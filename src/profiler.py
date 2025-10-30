import os, time, json, threading, subprocess
from functools import wraps
import torch

ROCM_SMI = os.environ.get("ROCM_SMI", "/opt/rocm/bin/rocm-smi")

import re

def _to_float(x):
    try:
        if isinstance(x, (int, float)): return float(x)
        if isinstance(x, str):
            m = re.findall(r"[-+]?\d*\.?\d+", x)
            return float(m[0]) if m else None
    except Exception:
        return None

def _to_int(x):
    try:
        if isinstance(x, int): return x
        if isinstance(x, float): return int(x)
        if isinstance(x, str):
            m = re.findall(r"\d+", x)
            return int(m[0]) if m else None
    except Exception:
        return None

def _first(d, keys):
    for k in keys:
        if k in d: return d[k]
    return None

def _rocm_smi_snapshot():
    try:
        out = subprocess.check_output(
            [ROCM_SMI, "--showtemp", "--showuse", "--showclocks",
             "--showmeminfo", "vram", "--showpower", "--json"],
            stderr=subprocess.DEVNULL, text=True
        )
        data = json.loads(out)
        flat = {"ts": time.time(), "raw": data}

        card = next(iter(data.values())) if isinstance(data, dict) else data[0]

        temp = _first(card, [
            "Temperature (Sensor junction) (C)",
            "Temperature (Sensor edge) (C)",
            "Temperature (Sensor die) (C)",
            "Temperature (C)",
        ])
        gpu_use = _first(card, ["GPU use (%)", "GPU use %"])
        mem_use = _first(card, ["Memory use (%)", "Memory use %"])

        sclk = _first(card, ["sclk clock speed:", "sclk clock speed (MHz)", "sclk"])
        mclk = _first(card, ["mclk clock speed:", "mclk clock speed (MHz)", "mclk"])

        vram_total_b = _first(card, ["VRAM Total Memory (B)"])
        vram_used_b  = _first(card, ["VRAM Used Memory (B)", "VRAM Total Used Memory (B)"])

        power = _first(card, [
            "Average Graphics Package Power (W)",
            "Average GPU Power (W)",
            "Power (W)",
        ])

        flat.update({
            "gpu_busy_percent": _to_float(gpu_use),
            "mem_busy_percent": _to_float(mem_use),
            "sclk_mhz": _to_float(sclk),
            "mclk_mhz": _to_float(mclk),
            "temp_c": _to_float(temp),
            "power_w": _to_float(power),
            "vram_total_mb": (_to_int(vram_total_b) or 0) / 1048576 if vram_total_b is not None else None,
            "vram_used_mb":  (_to_int(vram_used_b)  or 0) / 1048576 if vram_used_b  is not None else None,
        })
        return flat
    except Exception:
        return {"ts": time.time(), "error": "rocm-smi-parse-failed"}

def gpu_profile(outfile="perf.jsonl", poll_interval=0.5):
    """
    Decorator: Samples ROCm + PyTorch stats while fn runs, writes JSONL.
    Each line: {"phase": "sample"|"summary", ...}
    """
    os.makedirs(os.path.dirname(outfile) or ".", exist_ok=True)

    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            samples = []
            stop = threading.Event()

            def sampler():
                while not stop.is_set():
                    snap = _rocm_smi_snapshot()
                    if torch.cuda.is_available():
                        snap.update({
                            "torch_mem_alloc_mb": torch.cuda.memory_allocated(0)/1048576,
                            "torch_mem_reserved_mb": torch.cuda.memory_reserved(0)/1048576,
                            "torch_max_mem_alloc_mb": torch.cuda.max_memory_allocated(0)/1048576,
                        })
                    samples.append({"phase": "sample", **snap})
                    time.sleep(poll_interval)
            
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()

            t0 = time.time()
            thr = threading.Thread(target=sampler, daemon=True)
            thr.start()

            try:
                result = fn(*args, **kwargs)
            finally:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                stop.set(); thr.join(timeout=2.0)
                t1 = time.time()

                summary = {
                    "phase": "summary",
                    "fn": fn.__name__,
                    "wall_time_s": t1-t0,
                    "torch_max_mem_alloc_mb": torch.cuda.max_memory_allocated(0)/1048576 if torch.cuda.is_available() else None,
                    "timestamp_end": t1,
                }

                with open(outfile, "a") as f:
                    for s in samples: f.write(json.dumps(s) + "\n")
                    f.write(json.dumps(summary) + "\n")
            return result
        return wrapper
    return decorator