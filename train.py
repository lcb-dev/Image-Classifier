import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchvision import datasets
import torchvision.transforms as transforms

from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

from src import SmallCifarNet, train_one_epoch, evaluate, gpu_profile, gpu_device

# Define dataset(s)!
# Will be using the CIFAR100 image recognition data set for training. 

# --- transforms & datasets  --- #
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
])
train_ds = datasets.CIFAR100(root='data', train=True,  download=True, transform=transform)
test_ds  = datasets.CIFAR100(root='data', train=False, download=True, transform=transform)

# --- loaders --- #
train_loader = DataLoader(train_ds, batch_size=128, shuffle=True,  num_workers=4, pin_memory=True)
val_loader   = DataLoader(test_ds,  batch_size=256, shuffle=False, num_workers=4, pin_memory=True)

# --- device & model --- #
device = gpu_device()
if torch.cuda.is_available() and getattr(torch.version, "hip", None):
    print("Using AMD GPU via ROCm/HIP:", torch.cuda.get_device_name(0))

model = SmallCifarNet(num_classes=100).to(device)
opt = optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-2)
scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
os.makedirs("models", exist_ok=True)

# --- train loop --- #

@gpu_profile(outfile="perf/train_epoch.jsonl", poll_interval=0.5)
def profiled_train_epoch(*args, **kwargs):
    return train_one_epoch(*args, **kwargs)

@gpu_profile(outfile="perf/val_epoch.jsonl", poll_interval=0.5)
def profiled_evaluate(*args, **kwargs):
    return evaluate(*args, **kwargs)

best = 0.0
for epoch in range(15):
    tr = profiled_train_epoch(model, train_loader, opt, scaler, device)
    va = profiled_evaluate(model, val_loader, device)
    print(f"e{epoch+1}: train {tr['loss']:.3f}/{tr['acc']:.3f} | val {va['loss']:.3f} top1 {va['top1']:.3f} top5 {va['top5']:.3f}")
    if va["top1"] > best:
        best = va["top1"]
        torch.save(model.state_dict(), "models/smallcifarnet.pt")


# --- eval --- #
model.load_state_dict(torch.load("models/smallcifarnet.pt", map_location=device))
final = evaluate(model, val_loader, device)
print(f"FINAL top1 {final['top1']:.3f}, top5 {final['top5']:.3f}, loss {final['loss']:.3f}")

