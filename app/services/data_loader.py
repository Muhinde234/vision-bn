"""
Reads training.json / test.json and converts the malaria dataset to YOLO format.

Output layout
─────────────
malaria/yolo_dataset/
    images/
        train/   ← images from training.json
        val/     ← images from test.json
    labels/
        train/   ← one .txt per image
        val/
    data.yaml    ← Ultralytics dataset config
"""
import json
import shutil
from pathlib import Path
from typing import Dict

from app.utils.yolo_converter import CLASS_NAMES, object_to_yolo_line

# ── Path constants ────────────────────────────────────────────────────────────
_BACKEND_DIR = Path(__file__).resolve().parents[2]          # vision-backend/
MALARIA_DIR  = _BACKEND_DIR / "malaria"
IMAGES_SRC   = MALARIA_DIR / "images"
YOLO_DIR     = MALARIA_DIR / "yolo_dataset"
TRAIN_JSON   = MALARIA_DIR / "training.json"
TEST_JSON    = MALARIA_DIR / "test.json"


# ── Internal helpers ──────────────────────────────────────────────────────────

def _write_split(records: list, split: str) -> Dict:
    """
    Copy images and write YOLO label files for one dataset split.
    Returns per-split statistics.
    """
    img_out = YOLO_DIR / "images" / split
    lbl_out = YOLO_DIR / "labels" / split
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    stats = {"images": 0, "objects": 0, "skipped_images": 0, "skipped_objects": 0}

    for record in records:
        # Resolve source image (pathname starts with "/images/...")
        rel_path  = record["image"]["pathname"].lstrip("/")
        src_image = MALARIA_DIR / rel_path

        if not src_image.exists():
            stats["skipped_images"] += 1
            continue

        shape   = record["image"]["shape"]
        img_h   = shape["r"]   # rows    = height
        img_w   = shape["c"]   # columns = width
        stem    = src_image.stem
        suffix  = src_image.suffix.lower()

        # ── copy image (skip if already there) ────────────────────────────────
        dst_img = img_out / f"{stem}{suffix}"
        if not dst_img.exists():
            shutil.copy2(src_image, dst_img)
        stats["images"] += 1

        # ── write YOLO label file ─────────────────────────────────────────────
        lines = []
        for obj in record.get("objects", []):
            line = object_to_yolo_line(obj, img_h, img_w)
            if line:
                lines.append(line)
                stats["objects"] += 1
            else:
                stats["skipped_objects"] += 1

        lbl_path = lbl_out / f"{stem}.txt"
        lbl_path.write_text("\n".join(lines), encoding="utf-8")

    return stats


def _write_yaml() -> Path:
    """Generate data.yaml for Ultralytics."""
    yaml_path = YOLO_DIR / "data.yaml"
    names_block = "\n".join(f"  {i}: {n}" for i, n in enumerate(CLASS_NAMES))
    content = (
        f"path: {YOLO_DIR.as_posix()}\n"
        f"train: images/train\n"
        f"val:   images/val\n\n"
        f"nc: {len(CLASS_NAMES)}\n"
        f"names:\n{names_block}\n"
    )
    yaml_path.write_text(content, encoding="utf-8")
    return yaml_path


# ── Public API ────────────────────────────────────────────────────────────────

def prepare_dataset(force: bool = False) -> Dict:
    """
    Convert both JSON splits to YOLO format.

    Args:
        force: If True, re-process even if output already exists.

    Returns:
        Summary dict with per-split statistics and paths.
    """
    if not TRAIN_JSON.exists():
        raise FileNotFoundError(f"training.json not found at {TRAIN_JSON}")
    if not TEST_JSON.exists():
        raise FileNotFoundError(f"test.json not found at {TEST_JSON}")

    # Short-circuit if already prepared and force=False
    if not force and (YOLO_DIR / "data.yaml").exists():
        return {
            "status":  "already_prepared",
            "yaml":    str(YOLO_DIR / "data.yaml"),
            "classes": CLASS_NAMES,
        }

    with open(TRAIN_JSON, encoding="utf-8") as f:
        train_records = json.load(f)
    with open(TEST_JSON, encoding="utf-8") as f:
        test_records  = json.load(f)

    train_stats = _write_split(train_records, "train")
    val_stats   = _write_split(test_records,  "val")
    yaml_path   = _write_yaml()

    return {
        "status":  "prepared",
        "train":   train_stats,
        "val":     val_stats,
        "yaml":    str(yaml_path),
        "classes": CLASS_NAMES,
    }
