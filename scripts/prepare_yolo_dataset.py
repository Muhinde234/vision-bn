"""
Converts malaria JSON annotations → YOLO format dataset.

Output structure:
    yolo_dataset/
        images/train/  images/val/  images/test/
        labels/train/  labels/val/  labels/test/
        dataset.yaml

Run: python scripts/prepare_yolo_dataset.py
"""
import json
import random
import shutil
import sys
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parents[1]
MALARIA    = ROOT / "malaria"
IMAGES_SRC = MALARIA / "images"
OUT        = ROOT / "yolo_dataset"

# ── Class registry (no 'difficult' — skip those annotations) ──────────────────
CLASSES = [
    "red blood cell",   # 0 — healthy RBC (dominant background class)
    "trophozoite",      # 1 — parasite: active feeding stage
    "ring",             # 2 — parasite: earliest stage
    "schizont",         # 3 — parasite: mature, high-density
    "gametocyte",       # 4 — parasite: sexual stage, transmissible
    "leukocyte",        # 5 — white blood cell (non-parasite)
]
CLS2ID = {c: i for i, c in enumerate(CLASSES)}
SKIP   = {"difficult"}


# ── Helpers ───────────────────────────────────────────────────────────────────

def bbox_to_yolo(bb: dict, h: int, w: int) -> tuple[float, float, float, float]:
    r1, c1 = bb["minimum"]["r"], bb["minimum"]["c"]
    r2, c2 = bb["maximum"]["r"], bb["maximum"]["c"]
    if r2 <= r1 or c2 <= c1:
        return None
    cx = ((c1 + c2) / 2) / w
    cy = ((r1 + r2) / 2) / h
    bw = (c2 - c1) / w
    bh = (r2 - r1) / h
    return (
        min(max(cx, 0.0), 1.0),
        min(max(cy, 0.0), 1.0),
        min(max(bw, 0.001), 1.0),
        min(max(bh, 0.001), 1.0),
    )


def write_split(items: list, split: str) -> int:
    img_dir = OUT / "images" / split
    lbl_dir = OUT / "labels" / split
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    for item in items:
        rel      = item["image"]["pathname"].lstrip("/")   # "images/xxx.png"
        src_path = MALARIA / rel
        if not src_path.exists():
            continue

        shape = item["image"]["shape"]
        ih, iw = shape["r"], shape["c"]

        lines = []
        for obj in item.get("objects", []):
            cat = obj["category"]
            if cat in SKIP or cat not in CLS2ID:
                continue
            result = bbox_to_yolo(obj["bounding_box"], ih, iw)
            if result is None:
                continue
            cx, cy, bw, bh = result
            lines.append(f"{CLS2ID[cat]} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

        shutil.copy2(src_path, img_dir / src_path.name)
        (lbl_dir / (src_path.stem + ".txt")).write_text("\n".join(lines))
        written += 1

    return written


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Malaria → YOLO Dataset Converter")
    print("=" * 60)

    with open(MALARIA / "training.json") as f:
        all_train = json.load(f)
    with open(MALARIA / "test.json") as f:
        test_data = json.load(f)

    random.seed(42)
    random.shuffle(all_train)
    val_n      = max(1, int(len(all_train) * 0.1))
    val_data   = all_train[:val_n]
    train_data = all_train[val_n:]

    print(f"\nSplit  →  train: {len(train_data)}  val: {len(val_data)}  test: {len(test_data)}")
    print(f"Output →  {OUT}\n")

    for split, items in [("train", train_data), ("val", val_data), ("test", test_data)]:
        n = write_split(items, split)
        print(f"  [{split:5s}]  {n:4d} images written")

    # dataset.yaml
    yaml = f"""# Malaria parasite detection — YOLOv9 dataset
path: {OUT.as_posix()}
train: images/train
val:   images/val
test:  images/test

nc: {len(CLASSES)}
names: {CLASSES}
"""
    (OUT / "dataset.yaml").write_text(yaml)

    print(f"\nClass map:")
    for i, c in enumerate(CLASSES):
        print(f"  {i}: {c}")

    print(f"\ndataset.yaml  → {OUT / 'dataset.yaml'}")
    print("Done.  Now run:  python scripts/train_yolo.py")


if __name__ == "__main__":
    main()
