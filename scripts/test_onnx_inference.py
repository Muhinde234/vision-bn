#!/usr/bin/env python3
"""
Sanity-check ONNX inference locally using onnxruntime.
Usage:
    python scripts/test_onnx_inference.py --onnx models/best.onnx --image tests/sample.jpg --imgsz 320
"""
import argparse
import time
from PIL import Image
import numpy as np
import onnxruntime as ort

parser = argparse.ArgumentParser()
parser.add_argument("--onnx", default="models/best.onnx")
parser.add_argument("--image", required=True)
parser.add_argument("--imgsz", type=int, default=320)
args = parser.parse_args()

# Preprocess: resize, normalize, CHW, N
img = Image.open(args.image).convert("RGB")
img_resized = img.resize((args.imgsz, args.imgsz), Image.BILINEAR)
arr = np.array(img_resized, dtype=np.float32) / 255.0
arr = arr.transpose(2, 0, 1)
arr = np.expand_dims(arr, 0)

sess_options = ort.SessionOptions()
# enable graph optimizations
sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
session = ort.InferenceSession(args.onnx, sess_options=sess_options, providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name

# Run inference
start = time.monotonic()
outputs = session.run(None, {input_name: arr})
elapsed = (time.monotonic() - start) * 1000
print(f"ONNX inference time: {elapsed:.2f} ms")
print("Outputs shapes:")
for i, out in enumerate(outputs):
    print(f"  output[{i}]: {np.array(out).shape}")

# Note: postprocessing (decode + NMS) depends on model export details; this script only measures raw runtime and output shapes.
