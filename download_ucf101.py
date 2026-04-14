from pathlib import Path
from collections import Counter
import shutil
import sys
import kagglehub


out_dir = Path("/Users/rodrigopaganini/master/xai/project/Text-to-concept/dataset/ucf101").expanduser()

# Download latest version
path = kagglehub.dataset_download("matthewjansen/ucf101-action-recognition")

print("Path to dataset files:", path)

shutil.copytree(path, out_dir / Path(path).name, dirs_exist_ok=True)
