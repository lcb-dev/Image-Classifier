import json, pathlib
from dataclasses import dataclass
from typing import List, Dict, Any
import pandas as pd
import matplotlib.pyplot as plt
import math, glob
from PIL import Image
import os

@dataclass
class PerfData:
    samples: pd.DataFrame
    summary: pd.DataFrame

def load_jsonl(path: str) -> PerfData:
    rows: List[Dict[str, Any]] = []
    summaries: List[Dict[str, Any]] = []
    with open(path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            if obj.get("phase") == "summary": 
                summaries.append(obj)
            else:
                rows.append(obj)

    samples = pd.DataFrame(rows)
    if not samples.empty and "ts" in samples.columns:
        t0 = samples["ts"].min()
        samples["t_rel_s"] = samples["ts"] - t0
    summary = pd.DataFrame(summaries)
    return PerfData(samples=samples, summary=summary)

def concat_runs(paths: List[str]) -> PerfData:
    all_samples, all_summ = [], []
    for p in paths:
        d = load_jsonl(p)
        run_id = pathlib.Path(p).stem
        if not d.samples.empty:
            df = d.samples.copy()
            df["run_id"] = run_id
            all_samples.append(df)
        if not d.summary.empty:
            s = d.summary.copy()
            s["run_id"] = run_id
            all_summ.append(s)
    return PerfData(
        samples=pd.concat(all_samples, ignore_index=True) if all_samples else pd.DataFrame(),
        summary=pd.concat(all_summ, ignore_index=True) if all_summ else pd.DataFrame()
    )

def save_csv(perf: PerfData, outdir: str):
    pathlib.Path(outdir).mkdir(parents=True, exist_ok=True)
    if not perf.samples.empty:
        perf.samples.to_csv(f"{outdir}/samples.csv", index=False)
    if not perf.summary.empty:
        perf.summary.to_csv(f"{outdir}/summary.csv", index=False)

def _plot(df: pd.DataFrame, y: str, outpath: str, title: str):
    if df.empty or y not in df.columns or "t_rel_s" not in df.columns:
        print(f"[skip] missing data for {y}")
        return
    plt.figure(figsize=(9, 4))
    if "run_id" in df.columns:
        for run, g in df.groupby("run_id"):
            plt.plot(g["t_rel_s"], g[y], label=str(run))
    else:
        plt.plot(df["t_rel_s"], df[y])
    plt.xlabel("Time (s)")
    plt.ylabel(y.replace("_", " "))
    plt.title(title)
    if "run_id" in df.columns and df["run_id"].nunique() > 1:
        plt.legend()
    pathlib.Path(outpath).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()

def plot_core_metrics(perf: PerfData, outdir: str):
    s = perf.samples
    specs = [
        ("gpu_busy_percent", "GPU Busy (%)",         "gpu_busy_percent.png"),
        ("mem_busy_percent", "VRAM Busy (%)",        "mem_busy_percent.png"),
        ("sclk_mhz",         "Shader Clock (MHz)",   "sclk_mhz.png"),
        ("mclk_mhz",         "Memory Clock (MHz)",   "mclk_mhz.png"),
        ("temp_c",           "GPU Temperature (C)",  "temp_c.png"),
        ("power_w",          "GPU Power (W)",        "power_w.png"),
        ("vram_used_mb",     "VRAM Used (MB)",       "vram_used_mb.png"),
        ("torch_mem_alloc_mb",    "PyTorch Allocated (MB)",    "torch_mem_alloc_mb.png"),
        ("torch_mem_reserved_mb", "PyTorch Reserved (MB)",     "torch_mem_reserved_mb.png"),
        ("torch_max_mem_alloc_mb","PyTorch Max Alloc (MB)",    "torch_max_mem_alloc_mb.png"),
    ]
    for col, title, fname in specs:
        _plot(s, col, f"{outdir}/{fname}", title)

def summarize(perf: PerfData) -> pd.DataFrame:
    if perf.samples.empty:
        return pd.DataFrame()
    df = perf.samples
    if "run_id" not in df.columns:
        df = df.assign(run_id="run")

    agg_spec = {
        "gpu_busy_percent": ["median","max"],
        "mem_busy_percent": ["median","max"],
        "sclk_mhz": ["median","max"],
        "mclk_mhz": ["median","max"],
        "temp_c": ["median","max"],
        "power_w": ["median","max"],
        "vram_used_mb": ["median","max"],
        "torch_mem_alloc_mb": ["median","max"],
        "torch_mem_reserved_mb": ["median","max"],
        "torch_max_mem_alloc_mb": ["max"],
        "t_rel_s": ["max"],
    }
    agg_spec = {k:v for k,v in agg_spec.items() if k in df.columns}
    if not agg_spec:
        return pd.DataFrame()

    agg = df.groupby("run_id", dropna=False).agg(agg_spec)
    agg.columns = ["_".join([c for c in col if c]) for col in agg.columns]
    return agg.reset_index()


def make_montage(image_paths, out_path, cols=2, pad=12, bg=(255,255,255)):
    """
    Combine images into a grid (cols x rows) and save a single PNG.
    Images can be different sizes; each is centered in its cell.
    """
    paths = [p for p in image_paths if os.path.exists(p)]
    if not paths: 
        print("[montage] no images"); return

    imgs = [Image.open(p).convert("RGB") for p in paths]
    cols = max(1, cols)
    rows = math.ceil(len(imgs)/cols)

    cell_w = max(im.width for im in imgs)
    cell_h = max(im.height for im in imgs)

    W = cols*cell_w + (cols+1)*pad
    H = rows*cell_h + (rows+1)*pad
    canvas = Image.new("RGB", (W, H), bg)

    for i, im in enumerate(imgs):
        r, c = divmod(i, cols)
        x0 = pad + c*cell_w + (cell_w - im.width)//2
        y0 = pad + r*cell_h + (cell_h - im.height)//2
        canvas.paste(im, (x0, y0))

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    canvas.save(out_path, "PNG")
    print(f"[montage] wrote {out_path}")