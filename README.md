# Project setup

## Machine Specs
- CPU: Ryzen 5 5600x (6 cores, 12 threads)
- GPU: Radeon RX 9060 XT (16GB)
    - Supported ROCm versions: 7.0.2, 7.0.1/7.0.0
- RAM: 16GB DDR4 3600MT/s

## Versions used:

- PyTorch Version:  2.10.0.dev20251028+rocm7.0
- HIP Version (ROCm):  7.0.51831-7c9236b16
- GPU Available:  True
- GPU Device:  AMD Radeon Graphics


## First test
Will be using the CIFAR100 dataset. Using ToTensor(), the image is torch.FloatTensor [C, H, W]. 
I will then index the data.
Matplotlib will be used to visualize this data. (expects: [H, W, C]).

## Performance monitoring and metrics
I was interested to see how ROCm performed with the training data, so I setup a performance testing framework. 
I wanted to log some different metrics:
- GPU Usage (%)
- GPU Power Usage (W)
- Shader Clock (MHz)
- GPU Temp (C)
- PyTorch Memory Allocation metrics (MB)
- VRAM Usage (MB)

To do this, I used a decorator "gpu_profile()" which ran some performance profiling code on a separate thread. 
It timed the process duration, and the metrics gathered were split into two parts:
 - PyTorch Metrics:
   - "torch_mem_alloc_mb"
   - "torch_mem_reserved_mb"
   - "torch_max_mem_alloc_mb"
 - GPU (ROCm) Metrics:
   - Clocks, power usage, shader clock, etc.

The ROCm profiling part was done using "/opt/rocm/bin/rocm-smi", which is just a CLI tool to monitor and manage AMD GPUs using the ROCm stack.

This initial performance data was then output into ".jsonl" files:
 - train_epoch.jsonl
 - val_epoch.jsonl

### Visualising the data
Next step was to visualise this data.
So, I created a new utility script perf_viz.py. 
This script used matplotlib, PIL, and pandas, to load process and then create a visual output of the data in the jsonl files.

This creates a separate png file of a graph for each metric measured.

One last improvement made here, was to create another quick CLI script to collate these images into a single png.
See: /tools/montage_perf.py

This then gave a clear view of metrics for the run.
