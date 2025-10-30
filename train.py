import os
import pathlib
import shutil
import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets
import torchvision.transforms as transforms

from src import SmallCifarNet, train_one_epoch, evaluate, gpu_profile, gpu_device

# CIFAR100 Normalization stats
MEAN=(0.5071,0.4865,0.4409); STD=(0.2673,0.2564,0.2762)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, default="data")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-train", type=int, default=128)
    p.add_argument("--batch-val", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--wd", type=float, default=1e-2)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--models-dir", type=str, default="models")
    p.add_argument("--perf-dir", type=str, default="perf")
    p.add_argument("--clear-perf", action="store_true", help="Delete existing perf logs before training")
    return p.parse_args()

def clear_dir_safe(dir_path: str):
    p = pathlib.Path(dir_path).resolve()
    if not p.exists():
        return
    if str(p) in ("/", str(pathlib.Path("/").resolve())):
        raise RuntimeError("Refusing to clear root directory.")
    for child in p.iterdir():
        if child.is_file() or child.is_symlink():
            child.unlink(missing_ok=True)
        elif child.is_dir():
            shutil.rmtree(child)

def make_transforms():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)  # OLD: (0.5,0.5,0.5), (0.5,0.5,0.5)
    ])

def make_datasets(data_root, transform):
    train_ds = datasets.CIFAR100(root=data_root, train=True,  download=True, transform=transform)
    test_ds  = datasets.CIFAR100(root=data_root, train=False, download=True, transform=transform)
    return train_ds, test_ds

def make_loaders(train_ds, val_ds, b_train, b_val, n_workers):
    train_loader = DataLoader(train_ds, batch_size=b_train, shuffle=True,
                              num_workers=n_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,  batch_size=b_val, shuffle=False,
                              num_workers=n_workers, pin_memory=True)
    return train_loader, val_loader

def make_model_and_opt(device, lr, wd):
    model = SmallCifarNet(num_classes=100).to(device)
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    use_amp = torch.cuda.is_available()
    scaler = torch.amp.GradScaler(device="cuda", enabled=use_amp)
    return model, opt, scaler

def make_profiled_fns(perf_dir):
    os.makedirs(perf_dir, exist_ok=True)

    @gpu_profile(outfile=os.path.join(perf_dir, "train_epoch.jsonl"), poll_interval=0.5)
    def profiled_train_epoch(*args, **kwargs):
        return train_one_epoch(*args, **kwargs)

    @gpu_profile(outfile=os.path.join(perf_dir, "val_epoch.jsonl"), poll_interval=0.5)
    def profiled_evaluate(*args, **kwargs):
        return evaluate(*args, **kwargs)

    return profiled_train_epoch, profiled_evaluate

def train(model, train_loader, val_loader, device, opt, scaler, epochs, models_dir, prof_tr, prof_ev):
    os.makedirs(models_dir, exist_ok=True)
    best = 0.0
    ckpt_path = os.path.join(models_dir, "smallcifarnet.pt")

    for epoch in range(epochs):
        tr = prof_tr(model, train_loader, opt, scaler, device)
        va = prof_ev(model, val_loader, device)
        print(f"e{epoch+1}: train {tr['loss']:.3f}/{tr['acc']:.3f} | "
              f"val {va['loss']:.3f} top1 {va['top1']:.3f} top5 {va['top5']:.3f}")
        if va["top1"] > best:
            best = va["top1"]
            torch.save(model.state_dict(), ckpt_path)

    return ckpt_path

def final_eval(model, ckpt_path, val_loader, device):
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    final = evaluate(model, val_loader, device)
    print(f"FINAL top1 {final['top1']:.3f}, top5 {final['top5']:.3f}, loss {final['loss']:.3f}")
    return final

def main():
    args = parse_args()

    if args.clear_perf:
        os.makedirs(args.perf_dir, exist_ok=True)
        clear_dir_safe(args.perf_dir)
        print(f"Cleared perf data in: {args.perf_dir}")

    device = gpu_device()
    if torch.cuda.is_available() and getattr(torch.version, "hip", None):
        print("Using AMD GPU via ROCm/HIP:", torch.cuda.get_device_name(0))

    tfm = make_transforms()
    train_ds, val_ds = make_datasets(args.data, tfm)
    train_loader, val_loader = make_loaders(train_ds, val_ds, args.batch_train, args.batch_val, args.num_workers)
    model, opt, scaler = make_model_and_opt(device, args.lr, args.wd)
    prof_tr, prof_ev = make_profiled_fns(args.perf_dir)

    ckpt = train(model, train_loader, val_loader, device, opt, scaler,
                 args.epochs, args.models_dir, prof_tr, prof_ev)

    final_eval(model, ckpt, val_loader, device)

if __name__ == "__main__":
    main()
