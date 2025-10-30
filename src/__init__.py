from .model import SmallCifarNet
from .engine import train_one_epoch, evaluate, topk_acc
from .profiler import gpu_profile

__all__ = [
    "SmallCifarNet",
    "train_one_epoch",
    "evaluate",
    "topk_acc",
    "gpu_profile",
    "gpu_device",
]

import torch

def gpu_device():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        if getattr(torch.version, "hip", None):
            print(f"Using AMD GPU via ROCm/HIP — {torch.cuda.get_device_name(0)}")
        else:
            print(f"Using NVIDIA CUDA — {torch.cuda.get_device_name(0)}")
    else:
        print("Using CPU")
    return dev

__version__ = "0.1.0"