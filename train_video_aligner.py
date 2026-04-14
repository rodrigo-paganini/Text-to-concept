import torch
import torchvision
import json
from pathlib import Path
import numpy as np
from tqdm import tqdm
from transformers import VideoMAEVideoProcessor, VideoMAEForVideoClassification, VideoMAEModel
from TextToConcept import TextToConcept
from video_utils import load_ssv2_split, load_k400_split, VideoMAETTCTWrapper, CTHWToTCHW, DivideBy255, \
    ToTensorTuple, SizedLabeledVideoDataset
from pytorchvideo.transforms import UniformTemporalSubsample, ApplyTransformToKey
from torchvision.transforms import Compose, Resize, CenterCrop
from pytorchvideo.data import UniformClipSampler


IMAGENET_MEAN = [0.485, 0.456, 0.406]  # TODO review values
IMAGENET_STD = [0.229, 0.224, 0.225]

SUBSET_NUM_SAMPLES = 20000
SEED=42


def get_device():
    return 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'


def main():
    np.random.seed(SEED)

    device = get_device()
    print("Using device:", device)

    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # feature_extractor = VideoMAEVideoProcessor.from_pretrained("MCG-NJU/videomae-small-finetuned-ssv2")  # TODO unnecessary??
    videomae_model = VideoMAEModel.from_pretrained("MCG-NJU/videomae-base")
    videomae_model = videomae_model.to(device)

    model = VideoMAETTCTWrapper(videomae_model, normalizer=torchvision.transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD))

    text_to_concept = TextToConcept(model, 'videomae', 'video')
    preprocessing_without_normalization = Compose([
        ApplyTransformToKey(
            key="video",
            transform=Compose([
                UniformTemporalSubsample(16),
                DivideBy255(),
                Resize((224, 224)),
                CenterCrop(224),
                CTHWToTCHW(),
                # Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]),
        ),
        ToTensorTuple(['video', 'label', 'video_name']),
    ])

    # SSV2_ROOT = Path("../dataset/ssv2")
    # LABELS_DIR = SSV2_ROOT / "labels"  # contains train.json, validation.json, test.json, labels
    K400_ROOT = Path("../dataset/k400")

    labeled_video_paths = load_k400_split(
        "train",
        K400_ROOT / "train/1",
    )
    clip_sampler = UniformClipSampler(clip_duration=3.0)

    # indices = np.random.choice(len(labeled_video_paths), size=SUBSET_NUM_SAMPLES, replace=False)
    # subset_paths = [labeled_video_paths[i] for i in indices]

    dset = SizedLabeledVideoDataset(
        video_sampler=torch.utils.data.SequentialSampler,
        # labeled_video_paths=subset_paths,
        labeled_video_paths=labeled_video_paths,
        clip_sampler=clip_sampler,
        transform=preprocessing_without_normalization,
    )
    print(len(dset))
    text_to_concept.train_linear_aligner(dset, batch_size=16, load_reps=False, save_dir='data/videomae_base/representations_k400')
    text_to_concept.save_linear_aligner('pretrained_aligners/videomae_base_aligner_20k_k400.pth')


if __name__ == '__main__':
    main()
