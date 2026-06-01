"""
Evaluate the trained malaria YOLOv8 model on the held-out test set.
Prints overall and per-class mAP50, mAP50-95, precision, and recall.
"""
import sys
from pathlib import Path
import tempfile
import yaml

ROOT = Path(__file__).resolve().parent
_CANDIDATE_PATHS = [
    ROOT / "models" / "best.pt",
    ROOT / "models" / "runs" / "malaria" / "weights" / "best.pt",
]
MODEL_PATH = next((p for p in _CANDIDATE_PATHS if p.exists()), _CANDIDATE_PATHS[0])
DATASET_ROOT = ROOT / "yolo_dataset"
CLASS_NAMES = [
    "red blood cell",
    "trophozoite",
    "ring",
    "schizont",
    "gametocyte",
    "leukocyte",
]


def main():
    if not MODEL_PATH.exists():
        print(f"ERROR: model not found at {MODEL_PATH}")
        print("Run training first via POST /train")
        sys.exit(1)

    test_images = DATASET_ROOT / "images" / "test"
    test_labels = DATASET_ROOT / "labels" / "test"
    if not test_images.exists() or not any(test_images.iterdir()):
        print(f"ERROR: no test images at {test_images}")
        sys.exit(1)

    print(f"Model  : {MODEL_PATH}")
    print(f"Test   : {test_images} ({len(list(test_images.glob('*')))} images)")
    print()

    # Build a temporary data.yaml that points val → test so model.val() runs on test
    data_cfg = {
        "path": str(DATASET_ROOT),
        "train": "images/train",
        "val":   "images/test",
        "nc":    len(CLASS_NAMES),
        "names": {i: n for i, n in enumerate(CLASS_NAMES)},
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml",
                                     delete=False, encoding="utf-8") as f:
        yaml.dump(data_cfg, f, allow_unicode=True)
        tmp_yaml = f.name

    from ultralytics import YOLO

    model = YOLO(str(MODEL_PATH))
    print("Running validation on test set …\n")

    results = model.val(
        data=tmp_yaml,
        split="val",     # 'val' key in yaml = test images
        imgsz=640,
        batch=16,
        device="cpu",
        verbose=False,
        plots=False,
        save_json=False,
    )

    # ── Overall metrics ────────────────────────────────────────────────────────
    mp   = float(results.box.mp)      # mean precision
    mr   = float(results.box.mr)      # mean recall
    map50    = float(results.box.map50)
    map5095  = float(results.box.map)

    print("=" * 55)
    print("  OVERALL (mean across all classes)")
    print("=" * 55)
    print(f"  Precision   : {mp:.4f}  ({mp*100:.1f}%)")
    print(f"  Recall      : {mr:.4f}  ({mr*100:.1f}%)")
    print(f"  mAP@50      : {map50:.4f}  ({map50*100:.1f}%)")
    print(f"  mAP@50-95   : {map5095:.4f}  ({map5095*100:.1f}%)")
    print()

    # ── Per-class metrics ──────────────────────────────────────────────────────
    ap50_per_class    = results.box.ap50      # shape (nc,)
    ap5095_per_class  = results.box.ap        # shape (nc,)
    p_per_class       = results.box.p         # shape (nc,)
    r_per_class       = results.box.r         # shape (nc,)

    print("=" * 55)
    print(f"  {'CLASS':<18} {'P':>6} {'R':>6} {'mAP50':>7} {'mAP50-95':>9}")
    print("-" * 55)
    for i, name in enumerate(CLASS_NAMES):
        p  = float(p_per_class[i])  if i < len(p_per_class)  else 0.0
        r  = float(r_per_class[i])  if i < len(r_per_class)  else 0.0
        a50 = float(ap50_per_class[i])   if i < len(ap50_per_class)   else 0.0
        a   = float(ap5095_per_class[i]) if i < len(ap5095_per_class) else 0.0
        print(f"  {name:<18} {p:>6.3f} {r:>6.3f} {a50:>7.3f} {a:>9.3f}")
    print("=" * 55)

    # ── Rating ────────────────────────────────────────────────────────────────
    print()
    if map50 >= 0.70:
        grade = "GOOD — ready for testing in a clinical demo"
    elif map50 >= 0.50:
        grade = "FAIR — usable but needs more training / data"
    else:
        grade = "POOR — model needs more epochs or data cleaning"
    print(f"  Assessment: {grade}")
    print()

    # Cleanup temp file
    Path(tmp_yaml).unlink(missing_ok=True)


if __name__ == "__main__":
    main()
