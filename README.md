# Project setup

## Machine Specs
- CPU: Ryzen 5 5600x (6 cores, 12 threads)
- GPU: Radeon RX 9060 XT (16GB)
    - Supported ROCm versions: 7.0.2, 7.0.1/7.0.0
- RAM: 16GB DDR4 3600MT/s

## Versions used:
-------------------  --------------------------
   PyTorch Version:  2.10.0.dev20251028+rocm7.0
HIP Version (ROCm):  7.0.51831-7c9236b16
CUDA Version field:
     GPU Available:  True
        GPU Device:  AMD Radeon Graphics
-------------------  --------------------------

## First test
Will be using the CIFAR100 dataset. Using ToTensor(), the image is torch.FloatTensor [C, H, W]. 
I will then index the data.
Matplotlib will be used to visualize this data. (expects: [H, W, C])