import argparse, torch
from PIL import Image
import torchvision.transforms as T
import torchvision.transforms.functional as F
from src import SmallCifarNet, gpu_device
from torchvision.datasets import CIFAR100

# CIFAR-100 stats (match train)
MEAN=(0.5071,0.4865,0.4409); STD=(0.2673,0.2564,0.2762)
CLASS_NAMES = CIFAR100(root="data", train=False, download=True).classes

BASE = T.Compose([T.Resize((32,32), antialias=True),
                  T.ToTensor(), T.Normalize(MEAN, STD)])

def load_model(device):
    m = SmallCifarNet(num_classes=100).to(device)
    m.load_state_dict(torch.load("models/smallcifarnet.pt", map_location=device))
    m.eval()
    return m

def make_tta_batch(img: Image.Image, mode: str) -> torch.Tensor:
    """Return a batch [N,3,32,32] according to TTA mode."""
    img = img.convert("RGB")
    if mode == "none":
        xs = [BASE(img)]
    elif mode == "flip":
        xs = [BASE(img), BASE(img.transpose(Image.FLIP_LEFT_RIGHT))]
    elif mode == "5crop":
        big = F.resize(img, 36, antialias=True)
        crops = T.FiveCrop(32)(big) 
        xs = [T.Normalize(MEAN, STD)(T.ToTensor()(c)) for c in crops]
    elif mode == "flip5crop":
        big = F.resize(img, 36, antialias=True)
        crops = list(T.FiveCrop(32)(big))
        flips = [c.transpose(Image.FLIP_LEFT_RIGHT) for c in crops]
        allv = crops + flips
        xs = [T.Normalize(MEAN, STD)(T.ToTensor()(c)) for c in allv]
    elif mode == "scale3":
        sizes = [28, 32, 36]
        xs = []
        for s in sizes:
            r = F.resize(img, s, antialias=True)
            cc = T.CenterCrop(32)(r)
            xs.append(T.Normalize(MEAN, STD)(T.ToTensor()(cc)))
    else:
        raise ValueError(f"Unknown TTA mode: {mode}")
    return torch.stack(xs, dim=0) 

def predict(img_path: str, topk: int, tta: str):
    device = gpu_device()
    model = load_model(device)
    img = Image.open(img_path)
    batch = make_tta_batch(img, tta).to(device)
    with torch.no_grad(), torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
        logits = model(batch).mean(dim=0, keepdim=True)
        probs = torch.softmax(logits, dim=1)[0]
        p, idx = probs.topk(topk)
    return [(CLASS_NAMES[i], float(pj)*100) for i, pj in zip(idx.tolist(), p.tolist())]

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Predict CIFAR-100 class for an image.")
    ap.add_argument("image", help="Path to image")
    ap.add_argument("--topk", type=int, default=5, help="How many guesses to show")
    ap.add_argument("--tta", choices=["none","flip","5crop","flip5crop","scale3"],
                    default="flip", help="Test-Time Augmentation mode")
    args = ap.parse_args()

    results = predict(args.image, args.topk, args.tta)
    for r, (lbl, pct) in enumerate(results, 1):
        print(f"{r}. {lbl} ({pct:.1f}%)")
