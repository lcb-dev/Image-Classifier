import torch
from torchvision import datasets
from tabulate import tabulate

is_cuda_available: bool = torch.cuda.is_available()
data_table = [
    ["PyTorch Version: ", torch.__version__], 
    ["HIP Version (ROCm): ", getattr(torch.version, "hip", None)],
    ["CUDA Version field: ", getattr(torch.version, "cuda", None)], 
    ["GPU Available: ", torch.cuda.is_available()],
    ["GPU Device: ", torch.cuda.get_device_name()]
]


def init_check():
    print(tabulate(data_table, colalign=("right", "left")))
    if(torch.cuda.is_available()):
        print(torch.cuda.get_device_name())



if __name__ == '__main__':
    init_check()