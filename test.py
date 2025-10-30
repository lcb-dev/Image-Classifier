import torch
import torchvision.transforms as T
from torchvision import datasets
from torch.utils.data import DataLoader
from src import SmallCifarNet, evaluate, gpu_device

def main():
    device = gpu_device()
    tfm = T.Compose(
        [T.ToTensor(), 
        T.Normalize((0.5,)*3, (0.5,)*3)
        ])

    test_ds = datasets.CIFAR100(root="data", train=False, download=True, transform=tfm)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=4, pin_memory=True)

    model = SmallCifarNet(num_classes=100).to(device)
    model.load_state_dict(torch.load("models/smallcifarnet.pt", map_location=device))
    out = evaluate(model, test_loader, device)

    print(f"TEST top1={out['top1']:.3f}  top5={out['top5']:.3f}  loss={out['loss']:.3f}")

if __name__ == '__main__':
    main()