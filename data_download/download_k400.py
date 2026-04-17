"""
Script to download the Kinetics-400 dataset (5% subset for training, and the full validation set for testing) from Kaggle, check for overlaps, and copy the files to a local directory.

Authors: E. Cabalé, H. Naranjo, R. Paganini
"""
from pathlib import Path
from collections import Counter
import shutil
import sys
import kagglehub


def scan_dir(root):
    vids = [p for p in Path(root).rglob("*") if p.suffix.lower() in EXTS]
    rels = {str(p.relative_to(root)) for p in vids}
    stems = {p.stem for p in vids}
    labels = Counter(
        p.relative_to(root).parts[0] for p in vids if len(p.relative_to(root).parts) >= 2
    )
    return len(vids), rels, stems, labels


def scan_txt(txt_path):
    rows = [
        line.strip().rsplit(None, 1)
        for line in Path(txt_path).read_text().splitlines()
        if line.strip()
    ]
    rels = {name for name, _ in rows}
    stems = {Path(name).stem for name, _ in rows}
    labels = Counter(label for _, label in rows)
    return len(rows), rels, stems, labels


# # Download train set
train_path = kagglehub.dataset_download("rohanmallick/kinetics-train-5per")
print("Path to dataset files:", train_path)

# # Download test set
test_path = kagglehub.dataset_download("ipythonx/k4testset")
print("Path to dataset files:", test_path)

# Check for overlaps
EXTS = {".mp4", }  # ".avi", ".mkv", ".webm", ".mov"}

out_dir = Path("dataset/k400").expanduser()
train_dir = Path(train_path) / "kinetics400_5per/kinetics400_5per/train"
test_txt = Path(test_path) / "kinetics400_val_list_videos.txt"
n_train, train_rels, train_stems, train_labels = scan_dir(train_dir)
n_test, test_rels, test_stems, test_labels = scan_txt(test_txt)

print("Train set: ", list(train_stems)[:5])
print("Test set: ", list(test_stems)[:5])

print("train videos:", n_train, "| labels:", len(train_labels))
print("test videos :", n_test, "| labels:", len(test_labels))
print("exact overlaps:", len(train_rels & test_rels))
print("stem overlaps :", len(train_stems & test_stems))

if train_rels & test_rels:
    print("\nexample overlaps:")
    for x in sorted(train_rels & test_rels)[:10]:
        print(x)

# Copy files outside the .cache dir
if len(train_rels & test_rels) == 0:  # No overlap!
    print("Copying dataset")
    shutil.copytree(train_dir, out_dir / f"train/{train_path.name}", dirs_exist_ok=True)
    shutil.copytree(test_path, out_dir / f"test/{test_path.name}", dirs_exist_ok=True)
else:
    raise ValueError("Overlap between train and test set!")
