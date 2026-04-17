"""
Script to download the UCF-101 dataset from Kaggle, check for overlaps, and copy the files to a local directory.

Authors: E. Cabalé, H. Naranjo, R. Paganini
"""
from pathlib import Path
from collections import Counter
import shutil
import sys
import kagglehub


out_dir = Path("dataset/ucf101").expanduser()

# Download latest version
path = kagglehub.dataset_download("matthewjansen/ucf101-action-recognition")

print("Path to dataset files:", path)

shutil.copytree(path, out_dir / Path(path).name, dirs_exist_ok=True)
