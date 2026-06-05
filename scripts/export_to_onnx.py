#!/usr/bin/env python3
"""
Export a YOLOv9 PyTorch weights file to ONNX using ultralytics API.
Usage:
    python scripts/export_to_onnx.py --pt models/best.pt --onnx models/best.onnx --imgsz 320
"""
import argparse
from ultralytics import YOLO

parser = argparse.ArgumentParser()
parser.add_argument("--pt", default="models/best.pt", help="Path to .pt weights")
parser.add_argument("--onnx", default="models/best.onnx", help="Output ONNX path")
parser.add_argument("--imgsz", type=int, default=320, help="Export image size")
parser.add_argument("--opset", type=int, default=12, help="ONNX opset version")
parser.add_argument("--simplify", action="store_true", help="Run onnx-simplifier if available")
args = parser.parse_args()

print(f"Loading model from {args.pt}...")
model = YOLO(args.pt)
print(f"Exporting to ONNX -> {args.onnx} (imgsz={args.imgsz})")
# ultralytics export; returns exported path(s)
model.export(format="onnx", imgsz=args.imgsz, simplify=args.simplify, opset=args.opset, file=args.onnx)
print("Export finished.")
