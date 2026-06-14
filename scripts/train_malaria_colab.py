"""
Colab-friendly malaria YOLOv9 training script.

Typical Colab usage:
    from google.colab import drive
    drive.mount('/content/drive')
    !python scripts/train_malaria_colab.py \
        --zip-path /content/drive/MyDrive/malaria_yolo_dataset.zip \
        --dataset-dir /content/yolo_dataset \
        --output-dir /content/drive/MyDrive/malaria_model

The script unzips the dataset if needed, writes a Colab-safe data.yaml,
trains YOLOv9 with stronger settings for small-object detection, and copies
best.pt to the output directory.
"""
from __future__ import annotations

import argparse
import shutil
import sys
import zipfile
from pathlib import Path

import yaml

CLASS_NAMES = [
    "red blood cell",
    "trophozoite",
    "ring",
    "schizont",
    "gametocyte",
    "leukocyte",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train malaria YOLOv9 in Colab")
    parser.add_argument("--zip-path", required=True, help="Path to the dataset zip file")
    parser.add_argument("--dataset-dir", required=True, help="Directory to extract the dataset into")
    parser.add_argument("--output-dir", required=True, help="Directory where best.pt will be copied")
    parser.add_argument("--epochs", type=int, default=150, help="Training epochs")
    parser.add_argument("--batch", type=int, default=8, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=960, help="Training image size")
    parser.add_argument("--device", default="0", help="YOLO device, for example 0 or cpu")
    parser.add_argument("--workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--name", default="malaria", help="Ultralytics run name")
    return parser.parse_args()


def ensure_dataset(zip_path: Path, dataset_dir: Path) -> None:
    train_images = dataset_dir / "images" / "train"
    val_images = dataset_dir / "images" / "val"
    if train_images.exists() and val_images.exists():
        print(f"Dataset already present at {dataset_dir}")
        return

    if not zip_path.exists():
        raise FileNotFoundError(f"Dataset zip not found: {zip_path}")

    print(f"Extracting {zip_path} to {dataset_dir} ...")
    dataset_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as archive:
        archive.extractall(dataset_dir)
    print("Dataset extraction complete.")


def write_data_yaml(dataset_dir: Path) -> Path:
    data_cfg = {
        "path": str(dataset_dir),
        "train": "images/train",
        "val": "images/val",
        "nc": len(CLASS_NAMES),
        "names": {index: name for index, name in enumerate(CLASS_NAMES)},
    }
    yaml_path = dataset_dir / "data.yaml"
    with yaml_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data_cfg, handle, sort_keys=False, allow_unicode=True)
    return yaml_path


def count_split_images(dataset_dir: Path, split: str) -> int:
    images_dir = dataset_dir / "images" / split
    return len(list(images_dir.glob("*"))) if images_dir.exists() else 0


def train_model(yaml_path: Path, args: argparse.Namespace):
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("ultralytics is not installed. Run: pip install ultralytics") from exc

    model = YOLO("yolov9t.pt")
    return model.train(
        data=yaml_path.as_posix(),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        workers=args.workers,
        project=str(Path(args.output_dir).parent / "runs"),
        name=args.name,
        exist_ok=True,
        patience=35,
        save_period=10,
        cache=True,
        cos_lr=True,
        close_mosaic=15,
        optimizer="AdamW",
        lr0=0.002,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        rect=True,
        hsv_h=0.01,
        hsv_s=0.5,
        hsv_v=0.4,
        degrees=5.0,
        translate=0.05,
        scale=0.3,
        flipud=0.15,
        fliplr=0.5,
        mosaic=0.5,
        mixup=0.0,
        copy_paste=0.0,
    )


def main() -> int:
    args = parse_args()

    zip_path = Path(args.zip_path)
    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ensure_dataset(zip_path, dataset_dir)
    yaml_path = write_data_yaml(dataset_dir)

    train_count = count_split_images(dataset_dir, "train")
    val_count = count_split_images(dataset_dir, "val")
    print(f"Train images: {train_count}")
    print(f"Val images  : {val_count}")
    print(f"data.yaml    : {yaml_path}")

    results = train_model(yaml_path, args)

    best_src = Path(results.save_dir) / "weights" / "best.pt"
    best_dst = output_dir / "best.pt"
    if best_src.exists():
        shutil.copy2(best_src, best_dst)
        size_mb = best_dst.stat().st_size / 1_048_576
        print(f"Saved best model to: {best_dst} ({size_mb:.1f} MB)")
    else:
        raise FileNotFoundError(f"best.pt not found at {best_src}")

    if hasattr(results, "results_dict"):
        rd = results.results_dict
        print("\nValidation metrics")
        print(f"  Precision : {rd.get('metrics/precision(B)', 0):.4f}")
        print(f"  Recall    : {rd.get('metrics/recall(B)', 0):.4f}")
        print(f"  mAP@50    : {rd.get('metrics/mAP50(B)', 0):.4f}")
        print(f"  mAP@50-95 : {rd.get('metrics/mAP50-95(B)', 0):.4f}")

    print("\nNext steps")
    print("  1. Download best.pt from the output directory")
    print("  2. Compare rare-class recall against the previous run")
    print("  3. Retrain if schizont/gametocyte/leukocyte remain near zero")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
