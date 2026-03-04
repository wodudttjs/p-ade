#!/usr/bin/env python3
"""Quick GPU provider test"""
import sys, os
os.environ["PYTHONUNBUFFERED"] = "1"

# Method 1: torch preload
try:
    import torch
    print(f"torch cuda: {torch.version.cuda}, cudnn: {torch.backends.cudnn.version()}", flush=True)
except Exception as e:
    print(f"torch import failed: {e}", flush=True)

import onnxruntime as ort
print(f"ort version: {ort.__version__}", flush=True)
print(f"available: {ort.get_available_providers()}", flush=True)

sess = ort.InferenceSession(
    "models/rtmpose/yolox_l.onnx",
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
)
print(f"Active provider: {sess.get_providers()[0]}", flush=True)
