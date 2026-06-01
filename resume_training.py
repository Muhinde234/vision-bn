"""
Resume malaria YOLOv8n training from the last checkpoint.

Resumes from models/runs/malaria/weights/last.pt (epoch 12) and trains
to the original 50-epoch target. Copies best.pt to models/best.pt when done.

Usage:
    python resume_training.py
    python resume_training.py --device 0   # use first GPU if available
"""
import argparse
import shutil
import sys
from pathlib import Path

ROOT       = Path(__file__).resolve().parent
LAST_PT    = ROOT / "models" / "runs" / "malaria" / "weights" / "last.pt"
MODELS_DIR = ROOT / "models"
BEST_OUT   = MODELS_DIR / "best.pt"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cpu",
                   help="Device: 'cpu', '0' (first GPU). Default: cpu")
    return p.parse_args()


def main():
    args = parse_args()

    if not LAST_PT.exists():
        print(f"ERROR: checkpoint not found at {LAST_PT}")
        print("No previous training run to resume.")
        sys.exit(1)

    try:
        from ultralytics import YOLO
    except ImportError:
        print("ERROR: ultralytics not installed — run: pip install ultralytics")
        sys.exit(1)

    print("=" * 60)
    print("  Resuming malaria YOLOv8n training")
    print("=" * 60)
    print(f"  Checkpoint : {LAST_PT}")
    print(f"  Device     : {args.device}")
    print("  Resuming from epoch 12 → target epoch 50")
    print("  (estimated ~11 h on CPU, ~30 min on GPU)")
    print("=" * 60)
    print()

    model = YOLO(str(LAST_PT))

    results = model.train(resume=True)

    # Copy the best weights out to models/best.pt
    best_src = Path(results.save_dir) / "weights" / "best.pt"
    if best_src.exists():
        shutil.copy2(best_src, BEST_OUT)
        size_mb = BEST_OUT.stat().st_size / 1_048_576
        print(f"\n[OK] Best model → {BEST_OUT}  ({size_mb:.1f} MB)")
    else:
        # Fallback: copy last.pt if best.pt somehow missing
        last_src = Path(results.save_dir) / "weights" / "last.pt"
        if last_src.exists():
            shutil.copy2(last_src, BEST_OUT)
            print(f"[WARN] best.pt missing — copied last.pt to {BEST_OUT}")

    # Print final validation metrics
    print()
    print("=" * 60)
    print("  Final validation metrics")
    print("=" * 60)
    if hasattr(results, "results_dict"):
        rd = results.results_dict
        map50    = rd.get("metrics/mAP50(B)", 0)
        map5095  = rd.get("metrics/mAP50-95(B)", 0)
        prec     = rd.get("metrics/precision(B)", 0)
        rec      = rd.get("metrics/recall(B)", 0)
        print(f"  Precision  : {prec:.4f}")
        print(f"  Recall     : {rec:.4f}")
        print(f"  mAP@50     : {map50:.4f}")
        print(f"  mAP@50-95  : {map5095:.4f}")

    print()
    print("Next: run  python evaluate_model.py  to score on the test set.")


if __name__ == "__main__":
    main()
