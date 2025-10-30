import argparse, torch
from PIL import Image
import torchvision.transforms as T
from src import SmallCifarNet, gpu_device
from torchvision.datasets import CIFAR100

CLASS_NAMES = CIFAR100(root="data", train=False, download=True).classes

tfm = T.Compose([T.Resize((32, 32)), T.ToTensor(), T.Normalize((0.5,)*3, (0.5,)*3)])

def load_model(device):
    m = SmallCifarNet(num_classes=100).to(device)
    m.load_state_dict(torch.load("models/smallcifarnet.pt", map_location=device))
    m.eval()
    return m

def predict(path, topk=5):
    device = gpu_device()
    img = Image.open(path).convert("RGB")
    x=tfm(img).unsqueeze(0).to(device)
    with torch.no_grad(), torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
        logits = load_model(device)(x)
        probs = torch.softmax(logits, dim=1)[0]
        p, idx = probs.topk(topk)
    return [(CLASS_NAMES[i], float(pj)*100) for i, pj in zip(idx.tolist(), p.tolist())]

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("image"); ap.add_argument("--topk", type=int, default=5)
    args = ap.parse_args()
    for r, (lbl, pct) in enumerate(predict(args.image, args.topk), 1):
        print(f"{r}. {lbl} ({pct:.1f}%)")