import os, time, json, threading, subprocess
from functools import wraps
import torch

ROCM_SMI = os.environ.get("ROCM_SMI", "/opt/rocm/bin/rocm-smi")

def _rocm_smi_snapshot():
    try:
        out = subprocess.check_output(
            [ROCM_SMI, "--showtemp", "--showuse", "--showclocks",
            "--showmeminfo", "vram", "--showpower", "--json"],
            stderr=subprocess.DEVNULL, text=True)
        data = json.loads(out)

        def first(d, keys, default=None):
            for k in keys:
                if k in d: return d[k]
            return default

        flat = {"ts": time.time(), "raw": data}

        try:
            card = next(iter(data.values())) if isinstance(data, dict) else data[0]
            flat.update({
                "gpu_busy_percent": first(card, ["GPU use %", "GPU use (%)"]),
                "mem_busy_percent": first(card, ["Memory use %", "Memory use (%)"]),
                "sclk_mhz": first(card, ["sclk clock speed (MHz)", "sclk"]),
                "mclk_mhz": first(card, ["mclk clock speed (MHz)", "mclk"]),
                "temp_c": first(card, ["Temperature (Sensor die) (C)", "Temperature (Sensor edge) (C)", "Temperature (C)"]),
                "power_w": first(card, ["Average Graphics Package Power (W)", "Average GPU Power (W)", "Power (W)"]),
                "vram_total_mb": first(card, ["VRAM Total Memory (B)"], 0)/1048576 if first(card, ["VRAM Total Memory (B)"]) else None,
                "vram_used_mb": first(card, ["VRAM Used Memory (B)"], 0)/1048576 if first(card, ["VRAM Used Memory (B)"]) else None,
            })
        except Exception:
            pass
        return flat
    except Exception:
        return {"ts": time.time(), "error": "rocm-smi-unavailable"}

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