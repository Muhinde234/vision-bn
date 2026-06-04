import argparse
import os
import yaml
from PIL import Image
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score


def xywhn_to_xyxy(x, y, w, h, iw, ih):
    cx = x * iw
    cy = y * ih
    bw = w * iw
    bh = h * ih
    x1 = cx - bw / 2
    y1 = cy - bh / 2
    x2 = cx + bw / 2
    y2 = cy + bh / 2
    return [x1, y1, x2, y2]


def iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)
    inter = inter_w * inter_h
    a1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    a2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])
    union = a1 + a2 - inter
    return 0 if union == 0 else inter / union


def load_yaml(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def read_labels(label_path, iw, ih):
    gts = []
    if not os.path.exists(label_path):
        return gts
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls = int(parts[0])
            x, y, w, h = map(float, parts[1:5])
            gts.append((cls, xywhn_to_xyxy(x, y, w, h, iw, ih)))
    return gts


def match_detections_to_gts(dets, gts, iou_thresh=0.5):
    matches = []
    gt_matched = [False] * len(gts)
    det_matched = [False] * len(dets)
    idxs = sorted(range(len(dets)), key=lambda i: dets[i][1], reverse=True)
    for i in idxs:
        best_i = -1
        best_iou = 0
        for j, (gcls, gbox) in enumerate(gts):
            if gt_matched[j]:
                continue
            cur_iou = iou(dets[i][2], gbox)
            if cur_iou > best_iou:
                best_iou = cur_iou
                best_i = j
        if best_i != -1 and best_iou >= iou_thresh:
            matches.append((best_i, i))
            gt_matched[best_i] = True
            det_matched[i] = True
    return matches, gt_matched, det_matched


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', required=True)
    parser.add_argument('--data', required=True)
    parser.add_argument('--img', type=int, default=640)
    parser.add_argument('--out', default='runs/diagnostics')
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    data = load_yaml(args.data)
    val_images_dir = data.get('val') or data.get('test')
    if not val_images_dir:
        print('No val/test path found in data yaml')
        return

    model = YOLO(args.weights)

    names = data.get('names')
    if isinstance(names, dict):
        class_names = [names[i] for i in sorted(map(int, names.keys()))]
    elif isinstance(names, list):
        class_names = names
    else:
        class_names = [str(i) for i in range(100)]

    n_classes = len(class_names)

    per_class_detections = defaultdict(list)
    y_true = []
    y_pred = []

    # iterate images
    for root, _, files in os.walk(val_images_dir):
        for fname in sorted(files):
            if not fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                continue
            img_path = os.path.join(root, fname)
            im = Image.open(img_path)
            iw, ih = im.size
            label_fname = os.path.splitext(fname)[0] + '.txt'
            # guess label path: sibling labels directory
            labels_dir = os.path.join(os.path.dirname(val_images_dir).replace('images', 'labels'))
            label_path = os.path.join(labels_dir, label_fname)
            gts = read_labels(label_path, iw, ih)

            results = model.predict(source=img_path, imgsz=args.img, conf=0.001, iou=0.5, device='cpu')
            if len(results) == 0:
                dets = []
            else:
                r = results[0]
                boxes = r.boxes.xyxy.cpu().numpy() if hasattr(r, 'boxes') else np.array([])
                scores = r.boxes.conf.cpu().numpy() if hasattr(r, 'boxes') else np.array([])
                classes = r.boxes.cls.cpu().numpy().astype(int) if hasattr(r, 'boxes') else np.array([])
                dets = []
                for b, s, c in zip(boxes, scores, classes):
                    dets.append((int(c), float(s), [float(b[0]), float(b[1]), float(b[2]), float(b[3])]))

            matches, gt_matched, det_matched = match_detections_to_gts(dets, gts, iou_thresh=0.5)

            for gt_idx, det_idx in matches:
                gt_cls = gts[gt_idx][0]
                pred_cls = dets[det_idx][0]
                y_true.append(gt_cls)
                y_pred.append(pred_cls)

            for i, matched in enumerate(gt_matched):
                if not matched:
                    y_true.append(gts[i][0])
                    y_pred.append(-1)

            for i, matched in enumerate(det_matched):
                if not matched:
                    y_true.append(-1)
                    y_pred.append(dets[i][0])

            for cls in range(n_classes):
                dets_cls = [(i, s, box) for i, (c, s, box) in enumerate(dets) if c == cls]
                for i, s, box in dets_cls:
                    matched_same_class = False
                    for (gt_idx, det_idx) in matches:
                        if det_idx == i and gts[gt_idx][0] == cls:
                            matched_same_class = True
                            break
                    per_class_detections[cls].append((s, 1 if matched_same_class else 0))

    # confusion matrix
    cm = np.zeros((n_classes + 1, n_classes + 1), dtype=int)
    for t, p in zip(y_true, y_pred):
        ti = n_classes if t == -1 else t
        pi = n_classes if p == -1 else p
        cm[ti, pi] += 1

    # plot confusion matrix
    import seaborn as sns
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names + ['background'], yticklabels=class_names + ['background'], ax=ax)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix (last row/col = background)')
    fig.savefig(os.path.join(args.out, 'confusion_matrix.png'), bbox_inches='tight')
    plt.close(fig)

    # PR curves and AP
    ap_per_class = {}
    plt.figure(figsize=(8, 6))
    for cls in range(n_classes):
        items = per_class_detections.get(cls, [])
        if len(items) == 0:
            ap_per_class[class_names[cls]] = 0.0
            continue
        scores = np.array([x[0] for x in items])
        labels_bin = np.array([x[1] for x in items])
        try:
            ap = average_precision_score(labels_bin, scores)
        except Exception:
            ap = 0.0
        ap_per_class[class_names[cls]] = float(ap)
        precision, recall, _ = precision_recall_curve(labels_bin, scores)
        plt.plot(recall, precision, label=f'{class_names[cls]} (AP={ap:.3f})')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Per-class PR curves')
    plt.legend(loc='lower left')
    plt.grid(True)
    plt.savefig(os.path.join(args.out, 'pr_curves.png'), bbox_inches='tight')
    plt.close()

    with open(os.path.join(args.out, 'ap_per_class.yaml'), 'w') as f:
        yaml.safe_dump(ap_per_class, f)

    print('Diagnostics saved to', args.out)


if __name__ == '__main__':
    main()
