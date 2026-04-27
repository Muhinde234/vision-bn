"""
Converts the custom malaria bbox format → YOLO label format.

Input box  : {"minimum": {"r": y_min, "c": x_min}, "maximum": {"r": y_max, "c": x_max}}
Output line: "class_id  cx  cy  w  h"  (all values normalised 0-1)

Note: r = row = y-axis,  c = column = x-axis
"""
from typing import Dict, List, Optional, Tuple

# ── Class registry (must match scripts/prepare_yolo_dataset.py ordering) ──────
CLASS_NAMES: List[str] = [
    "red blood cell",   # 0  — healthy RBCs (dominant background class)
    "trophozoite",      # 1  — parasite: active feeding stage
    "ring",             # 2  — parasite: earliest infection stage
    "schizont",         # 3  — parasite: mature, high-density / high-risk
    "gametocyte",       # 4  — parasite: sexual stage, transmissible
    "leukocyte",        # 5  — white blood cell (non-parasite)
]

CLASS_MAP: Dict[str, int] = {name: idx for idx, name in enumerate(CLASS_NAMES)}


# ── Conversion helpers ────────────────────────────────────────────────────────

def bbox_to_yolo(
    row_min: int,
    col_min: int,
    row_max: int,
    col_max: int,
    img_height: int,
    img_width: int,
) -> Tuple[float, float, float, float]:
    """
    Convert pixel-space bounding box to YOLO normalised (cx, cy, w, h).
    Clamps all values to [0, 1] to guard against annotation overflow.
    """
    cx = ((col_min + col_max) / 2.0) / img_width
    cy = ((row_min + row_max) / 2.0) / img_height
    w  = (col_max - col_min) / img_width
    h  = (row_max - row_min) / img_height

    cx = min(max(cx, 0.0), 1.0)
    cy = min(max(cy, 0.0), 1.0)
    w  = min(max(w,  0.001), 1.0)
    h  = min(max(h,  0.001), 1.0)
    return cx, cy, w, h


def object_to_yolo_line(obj: dict, img_height: int, img_width: int) -> Optional[str]:
    """
    Convert one annotation object to a YOLO label line.
    Returns None if the category is not in CLASS_MAP.
    """
    category = obj.get("category", "")
    class_id = CLASS_MAP.get(category)
    if class_id is None:
        return None

    bb      = obj["bounding_box"]
    row_min = bb["minimum"]["r"]
    col_min = bb["minimum"]["c"]
    row_max = bb["maximum"]["r"]
    col_max = bb["maximum"]["c"]

    # Skip degenerate boxes
    if row_max <= row_min or col_max <= col_min:
        return None

    cx, cy, w, h = bbox_to_yolo(row_min, col_min, row_max, col_max, img_height, img_width)
    return f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"
