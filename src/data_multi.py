from dataclasses import dataclass
from typing import List, Tuple, Dict
import torch
from torch.utils.data import Dataset, ConcatDataset
from torchvision import datasets as tvd
import torchvision.transforms as T

IMAGENET_MEAN=(0.485,0.456,0.406); IMAGENET_STD=(0.229,0.224,0.225)

@dataclass
class DatasetSpec:
    name: str
    type: str 
    root: str
    split: str 

def build_transforms(size=224, train=True):
    if train:
        return T.Compose([
            T.RandomResizedCrop(size, scale=(0.6, 1.0)),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    else:
        return T.Compose([
            T.Resize(256), T.CenterCrop(size),
            T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

def make_one_dataset(spec: DatasetSpec, tfm):
    if spec.type == "cifar100":
        train = spec.split in ("train","training")
        return tvd.CIFAR100(root=spec.root, train=train, download=True, transform=tfm), 100
    if spec.type == "food101":
        return tvd.Food101(root=spec.root, split=spec.split, download=True, transform=tfm), 101
    if spec.type == "imagefolder":
        ds = tvd.ImageFolder(root=spec.root, transform=tfm)
        return ds, len(ds.classes)
    raise ValueError(f"Unknown type {spec.type}")

class RoutedConcat(Dataset):
    """
    Wraps multiple datasets; returns (image, class_idx, head_id).
    class_idx is re-indexed per head independently (0..C_head-1).
    head_id is an integer routing to the right classification head.
    """
    def __init__(self, named_subsets: List[Tuple[str, Dataset]]):
        self.name_to_head: Dict[str, int] = {}
        self.head_to_classes: Dict[int, List[str]] = {}
        self.datasets: List[Dataset] = []
        self.offsets: List[int] = []
        for hid, (name, ds) in enumerate(named_subsets):
            self.name_to_head[name] = hid
            self.datasets.append(_RelabelDataset(ds)) 
            self.head_to_classes[hid] = getattr(ds, "classes", [f"{name}_{i}" for i in range(9999)])
        cum = 0
        for d in self.datasets:
            self.offsets.append(cum)
            cum += len(d)
        self._len = cum

    def __len__(self): return self._len

    def __getitem__(self, idx):
        for d, off, name in zip(self.datasets, self.offsets, self.name_to_head.keys()):
            if idx < off + len(d):
                x, y = d[idx - off]
                head_id = self.name_to_head[name]
                return x, y, head_id
        raise IndexError

class _RelabelDataset(Dataset):
    """Ensure targets are contiguous 0..C-1 (many torchvision datasets already are)."""
    def __init__(self, base: Dataset):
        self.base = base
    def __len__(self): return len(self.base)
    def __getitem__(self, i):
        x, y = self.base[i]
        return x, int(y)
