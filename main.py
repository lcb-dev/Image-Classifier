import torch
from tabulate import tabulate

is_cuda_available: bool = torch.cuda.is_available()
data_table = [
    ["PyTorch Version: ", torch.__version__], 
    ["HIP Version (ROCm): ", getattr(torch.version, "hip", None)],
    ["CUDA Version field: ", getattr(torch.version, "cuda", None)], 
    ["GPU Available: ", torch.cuda.is_available()]
]


def init_check():
    print(tabulate(data_table, colalign=("right", "left")))


if __name__ == '__main__':
    init_check()