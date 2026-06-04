import argparse
import os
import random
import shutil
import yaml
from collections import defaultdict


def load_yaml(path):
    import yaml as _yaml
    with open(path, 'r') as f:
        return _yaml.safe_load(f)


def find_label_files(labels_dir):
    files = []
    for root, _, fnames in os.walk(labels_dir):
        for f in fnames:
            if f.lower().endswith('.txt'):
                files.append(os.path.join(root, f))
    return files


def parse_label_file(path):
    classes = []
    with open(path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            classes.append(int(parts[0]))
    return classes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True, help='data yaml')
    parser.add_argument('--target', type=int, default=500, help='target samples per class')
    parser.add_argument('--out', default=None, help='output augmented folder (defaults to yolo_dataset/augmented/train)')
    args = parser.parse_args()

    data = load_yaml(args.data)
    train_images = data.get('train')
    if not train_images:
        print('train path not found in data yaml')
        return

    # guess labels dir alongside images
    labels_dir = os.path.join(os.path.dirname(train_images).replace('images', 'labels'))
    if not os.path.isdir(labels_dir):
        # try sibling folder
        labels_dir = os.path.join(os.path.dirname(train_images), '..', 'labels')

    label_files = find_label_files(labels_dir)
    class_to_files = defaultdict(list)
    for lf in label_files:
        classes = parse_label_file(lf)
        for c in set(classes):
            class_to_files[c].append(lf)

    max_class = max(class_to_files.keys()) if class_to_files else -1
    print('Found classes:', sorted(class_to_files.keys()))

    out_dir = args.out or os.path.join(os.path.dirname(train_images), 'augmented', 'images')
    out_labels = os.path.join(os.path.dirname(train_images), 'augmented', 'labels')
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(out_labels, exist_ok=True)

    # copy existing train images and labels into augmented folder first
    # images dir may be nested; we will copy by filename
    for root, _, files in os.walk(train_images):
        for f in files:
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                src = os.path.join(root, f)
                dst = os.path.join(out_dir, f)
                if not os.path.exists(dst):
                    shutil.copy2(src, dst)
    for lf in label_files:
        fname = os.path.basename(lf)
        dst = os.path.join(out_labels, fname)
        if not os.path.exists(dst):
            shutil.copy2(lf, dst)

    # oversample
    for cls, files in class_to_files.items():
        cur = len(files)
        needed = max(0, args.target - cur)
        print(f'class {cls}: {cur} -> need {needed}')
        if needed <= 0:
            continue
        for i in range(needed):
            src_label = random.choice(files)
            base = os.path.splitext(os.path.basename(src_label))[0]
            # guess image with same base
            for ext in ('.jpg', '.jpeg', '.png'):
                src_img = os.path.join(os.path.dirname(train_images), base + ext)
                if os.path.exists(src_img):
                    break
            else:
                # try searching train_images
                src_img = None
                for root, _, fnames in os.walk(train_images):
                    for f in fnames:
                        if os.path.splitext(f)[0] == base:
                            src_img = os.path.join(root, f)
                            break
                    if src_img:
                        break
            if not src_img:
                continue
            new_base = f'{base}_dup_{i}'
            dst_img = os.path.join(out_dir, new_base + os.path.splitext(src_img)[1])
            dst_label = os.path.join(out_labels, new_base + '.txt')
            shutil.copy2(src_img, dst_img)
            shutil.copy2(src_label, dst_label)

    print('Augmented dataset created at', os.path.join(os.path.dirname(train_images), 'augmented'))


if __name__ == '__main__':
    main()
