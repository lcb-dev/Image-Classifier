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

## Project Structure
Currently this project consists of essentially three layers:
- Machine Learning parts
      - Data
      - Model
      - Training
- Performance Logging
      - Collect
      - Save
- Visualisation
      - Read logs
      - Plot data
      - Generate graphs

### Machine Learning Section
Train.py:
 - Parse CLI args
 - Builds everything (transforms, datasets, model, etc.)
 - Runs training and validation, for N epochs.
 - Saves the best model, and does a final evaluation
 - Calls profiled wrappers so each epoch gets measured.

PyTorch GPU API:
 - Despite using ROCm, the module is named "cuda", even on the ROCm builds.
 - Checks "torch.cuda.is_available()" before attempting to execute on the GPU.

Data pipeline (train.py):
 - Transforms: Normalization, makes training stable.
 - Datasets: Download and expose images+labels.
 - DataLoaders: Feeds tensors to the model

Model (src/model.py -> SmallCifarNet):
 - A from-scratch CNN.

Training engine (src/engine.py):
 - Loops over batches
 - runs the model in inference mode, computes loss, top-1 and top-5 accuracy.

### Performance logging
src/profiler.py:
 - While a wrapped function runs, samples metrics periodically and writes JSON lines to a file.
 - Metrics:
       - ROCm-smi: Temperature, clocks, power, GPU Utilisation, VRAM usage
       - PyTorch runtime: allocated/reserved/peak GPU memory.
 - perf/train_epoch.jsonl (samples during training epochs)
 - perf/val_epoch.jsonl (samples during evaluation)

Why "jsonl"?
 - One JSON object per line, so easy to stream, filter, and read into pandas.

### Visualisation
src/perf_viz.py:
 - load jsonl: Read one or many JSONL logs into DataFrames, compute a relative time column, and tag each file as a run.
 - plot core metrics: Create PNG time-series for present metrics. Skips any missing.
 - Summary: Outputs a small per-run table.
 - Montage creation: Stitch multiple PNGs into a single grid image.

Some CLI tools:
 - tools/plot_perf.py:
       - writes metric PNGs + CSVs + optional summary to perf/plots_train/.
       - Example command: "python -m tools.plot_perf perf/train_epoch.jsonl --out perf/plots_train"
 - tools/montage_perf.py:
       - Makes one combined image.
       - Example command: "python -m tools.montage_perf perf/plots_train --cols 3 --out perf/plots_train/montage.png".
   
## How it flows end-to-end
1. Start training:
       - python train.py --epochs N --clear-perf
       - Clears old logs
       - Builds data/model/optim/scaler
       - Best model saved to "models/..."
2. Plot & Inspect:
       - python -m tools.plot_perf perf/train_epoch.jsonl --out perf/plots_train
3. Reporting
       - Training prints epoch metrics
       - Final evaluation prints top-1/top-5 and loss.

## Cloning the repo
```bash
gh clone lcb-dev/Image-Classifier
```


## Actual test runs

### First run (5 epochs)
- python train.py --clear-perf --epochs 5
From this, I had an output (for each epoch) of:
```
e1: train 3.796/0.111 | val 3.285 top1 0.187 top5 0.471
e2: train 2.877/0.265 | val 2.755 top1 0.301 top5 0.613
e3: train 2.334/0.377 | val 2.470 top1 0.360 top5 0.676
e4: train 1.980/0.457 | val 2.023 top1 0.450 top5 0.769
e5: train 1.735/0.514 | val 1.867 top1 0.496 top5 0.799
FINAL top1 0.496, top5 0.799, loss 1.867
```

After running, I tested the model again using:
- python test.py
Which gave an output of:
```
TEST top1=0.496  top5=0.799  loss=1.867
```

From this, the Val loss from each epoch was trending downwards.
The top-1 and top-5 was also improving every epoch.

Since it was still improving, I decided to test again using 30 epochs. 
I will continue testing until improvement stops.

### Second run (30 epochs)
- python train.py --clear-perf --epochs 30
From this, I had an output of:
```
e1: train 3.710/0.123 | val 3.223 top1 0.203 top5 0.492
e2: train 2.841/0.272 | val 2.637 top1 0.305 top5 0.640
e3: train 2.338/0.376 | val 2.272 top1 0.401 top5 0.722
e4: train 1.994/0.455 | val 2.076 top1 0.442 top5 0.755
e5: train 1.754/0.512 | val 1.844 top1 0.492 top5 0.803
e6: train 1.548/0.563 | val 1.751 top1 0.522 top5 0.819
e7: train 1.384/0.601 | val 1.677 top1 0.534 top5 0.828
e8: train 1.230/0.641 | val 1.609 top1 0.561 top5 0.848
e9: train 1.101/0.676 | val 1.521 top1 0.582 top5 0.853
e10: train 0.965/0.711 | val 1.516 top1 0.586 top5 0.861
e11: train 0.849/0.743 | val 1.608 top1 0.577 top5 0.856
e12: train 0.748/0.770 | val 1.687 top1 0.568 top5 0.842
e13: train 0.645/0.800 | val 1.675 top1 0.581 top5 0.855
e14: train 0.562/0.823 | val 1.756 top1 0.574 top5 0.847
e15: train 0.492/0.844 | val 1.734 top1 0.585 top5 0.854
e16: train 0.430/0.863 | val 1.783 top1 0.589 top5 0.855
e17: train 0.384/0.877 | val 1.702 top1 0.600 top5 0.864
e18: train 0.333/0.894 | val 1.828 top1 0.589 top5 0.858
e19: train 0.305/0.903 | val 1.796 top1 0.595 top5 0.860
e20: train 0.269/0.916 | val 1.998 top1 0.578 top5 0.843
e21: train 0.260/0.917 | val 1.967 top1 0.580 top5 0.847
e22: train 0.242/0.923 | val 1.933 top1 0.598 top5 0.861
e23: train 0.222/0.929 | val 2.051 top1 0.579 top5 0.848
e24: train 0.210/0.933 | val 1.983 top1 0.597 top5 0.860
e25: train 0.207/0.933 | val 1.938 top1 0.591 top5 0.855
e26: train 0.182/0.943 | val 2.160 top1 0.584 top5 0.849
e27: train 0.180/0.943 | val 1.973 top1 0.596 top5 0.855
e28: train 0.176/0.944 | val 2.219 top1 0.579 top5 0.848
e29: train 0.159/0.951 | val 2.103 top1 0.587 top5 0.853
e30: train 0.162/0.949 | val 2.066 top1 0.592 top5 0.854
FINAL top1 0.600, top5 0.864, loss 1.702
```
After running, I tested the model again using:
- python test.py
```
TEST top1=0.600  top5=0.864  loss=1.702
```

So, what can be inferred from this, is that there was steady improvement up to about epoch 10, then it started overfitting.
The best Top-1 value came at epoch 17. 
- Best Top-1 (epoch 17): 0.600
- Lowest val loss (epoch 10): 1.516

Overall, not bad for a small model, but could certainly be improved.