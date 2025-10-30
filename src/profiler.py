import os, time, json, threading, subprocess
from functools import wraps
import torch

ROCM_SMI = os.environ.get("ROCM_SMI", "/opt/rocm/bin/rocm-smi")

def _rocm_smi_snapshot():
    pass