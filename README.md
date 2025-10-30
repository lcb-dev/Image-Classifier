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
