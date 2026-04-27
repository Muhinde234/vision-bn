"""
Train YOLOv9n on the malaria parasite dataset.

Prerequisites:
    pip install ultralytics
    python scripts/prepare_yolo_dataset.py   (run first)

Output:
    models/best.pt   — best checkpoint (auto-copied here after training)

Usage:
    python scripts/train_yolo.py
    python scripts/train_yolo.py --epochs 50 --batch 16 --device 0
"""
import argparse
import shutil
import sys
from pathlib import Path

ROOT        = Path(__file__).resolve().parents[1]
DATASET     = ROOT / "yolo_dataset" / "dataset.yaml"
MODELS_DIR  = ROOT / "models"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=30,
                   help="Training epochs (default 30; use 50+ for best accuracy)")
    p.add_argument("--batch",  type=int, default=8,
                   help="Batch size (reduce to 4 if out of memory)")
    p.add_argument("--imgsz",  type=int, default=640,
                   help="Input image size in pixels (default 640)")
    p.add_argument("--device", type=str, default="auto",
                   help="Device: 'auto', 'cpu', '0' (first GPU), '0,1' (multi-GPU)")
    p.add_argument("--workers", type=int, default=0,
                   help="DataLoader workers (0 = main process, safe on Windows)")
    return p.parse_args()


def resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    try:
        import torch
        return "0" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def main():
    args = parse_args()

    if not DATASET.exists():
        print(f"[ERROR] Dataset not found: {DATASET}")
        print("Run:  python scripts/prepare_yolo_dataset.py  first.")
        sys.exit(1)

    try:
        from ultralytics import YOLO
    except ImportError:
        print("[ERROR] ultralytics not installed.")
        print("Run:  pip install ultralytics")
        sys.exit(1)

    device = resolve_device(args.device)
    MODELS_DIR.mkdir(exist_ok=True)

    print("=" * 60)
    print("  YOLOv9n  —  Malaria Parasite Detection")
    print("=" * 60)
    print(f"  Dataset  : {DATASET}")
    print(f"  Epochs   : {args.epochs}")
    print(f"  Batch    : {args.batch}")
    print(f"  Img size : {args.imgsz}")
    print(f"  Device   : {device}")
    print("=" * 60)

    model = YOLO("yolov9n.pt")   # ~4 MB pretrained weights downloaded automatically

    results = model.train(
        data     = str(DATASET),
        epochs   = args.epochs,
        imgsz    = args.imgsz,
        batch    = args.batch,
        device   = device,
        workers  = args.workers,
        project  = str(ROOT / "runs" / "train"),
        name     = "malaria_yolov9n",
        exist_ok = True,
        patience = 15,          # early-stop if no improvement for 15 epochs
        save     = True,
        cache    = False,       # set True to cache images in RAM (needs ~8 GB)
        verbose  = True,
        # Augmentation (good defaults for microscopy images)
        hsv_h    = 0.01,        # minimal hue shift (blood smears are colour-consistent)
        hsv_s    = 0.5,
        hsv_v    = 0.4,
        degrees  = 15,          # rotation
        fliplr   = 0.5,
        flipud   = 0.3,
        mosaic   = 0.8,
    )

    # ── Copy best.pt ──────────────────────────────────────────────────────────
    best_src = Path(results.save_dir) / "weights" / "best.pt"
    best_dst = MODELS_DIR / "best.pt"

    if best_src.exists():
        shutil.copy2(best_src, best_dst)
        size_mb = best_dst.stat().st_size / 1_048_576
        print(f"\n[OK] Best model  → {best_dst}  ({size_mb:.1f} MB)")
    else:
        print(f"\n[WARN] best.pt not found at {best_src}")
        last_src = Path(results.save_dir) / "weights" / "last.pt"
        if last_src.exists():
            shutil.copy2(last_src, best_dst)
            print(f"[OK] last.pt     → {best_dst}")

    # ── Export to ONNX (lightweight for Render deployment) ────────────────────
    print("\nExporting to ONNX …")
    best_model = YOLO(str(best_dst))
    export_path = best_model.export(
        format   = "onnx",
        imgsz    = args.imgsz,
        dynamic  = False,       # fixed shape = faster onnxruntime inference
        simplify = True,        # graph simplification
        opset    = 17,
    )
    onnx_src = Path(str(export_path))
    onnx_dst = MODELS_DIR / "best.onnx"
    if onnx_src.exists():
        shutil.copy2(onnx_src, onnx_dst)
        onnx_mb = onnx_dst.stat().st_size / 1_048_576
        print(f"[OK] ONNX model  → {onnx_dst}  ({onnx_mb:.1f} MB)")
    else:
        print(f"[WARN] ONNX export path not found: {onnx_src}")

    # ── Validation metrics ────────────────────────────────────────────────────
    print("\n── Validation Results ─────────────────────────")
    metrics = model.val(data=str(DATASET))
    print(f"  mAP@0.50       : {metrics.box.map50:.4f}")
    print(f"  mAP@0.50:0.95  : {metrics.box.map:.4f}")
    print(f"  Precision      : {metrics.box.mp:.4f}")
    print(f"  Recall         : {metrics.box.mr:.4f}")
    print("───────────────────────────────────────────────")
    print("\n── Next steps ─────────────────────────────────")
    print("  1. git add models/best.onnx models/best.pt")
    print("  2. git commit -m 'feat: add trained YOLOv9 malaria model'")
    print("  3. git push")
    print("  Render will redeploy and automatically use the real model.")
    print("───────────────────────────────────────────────")


if __name__ == "__main__":
    main()
