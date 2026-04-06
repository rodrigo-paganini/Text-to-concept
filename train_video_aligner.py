import torch
import torchvision
import json
from pathlib import Path
import numpy as np
from tqdm import tqdm
from transformers import VideoMAEVideoProcessor, VideoMAEForVideoClassification
from TextToConcept import TextToConcept
from video_utils import make_dataset, VideoMAETTCTWrapper, CTHWToTCHW, DivideBy255, ToTensorTuple
from pytorchvideo.transforms import UniformTemporalSubsample, ApplyTransformToKey
from torchvision.transforms import Compose, Resize, CenterCrop
from pytorchvideo.data import LabeledVideoDataset, UniformClipSampler


IMAGENET_MEAN = [0.485, 0.456, 0.406]  # TODO review values
IMAGENET_STD = [0.229, 0.224, 0.225]

SUBSET_NUM_SAMPLES = 20000
SEED=42

def get_device():
    return 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'

class SizedLabeledVideoDataset(LabeledVideoDataset):
    def __len__(self):
        return len(self._labeled_videos)

def main():
    np.random.seed(SEED)

    device = get_device()
    print("Using device:", device)

    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # feature_extractor = VideoMAEVideoProcessor.from_pretrained("MCG-NJU/videomae-small-finetuned-ssv2")  # TODO unnecessary??
    videomae_model = VideoMAEForVideoClassification.from_pretrained("MCG-NJU/videomae-small-finetuned-ssv2")
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
        ToTensorTuple(['video', 'label']),
    ])

    # # loading imagenet dataset to train aligner.
    # dset = torchvision.datasets.ImageNet(
    #     # root='/fs/cml-datasets/ImageNet/ILSVRC2012',
    #     root='.cache/kagglehub/datasets/ambityga/imagenet100',
    #     split='train',
    #     transform=preprocessing_without_normalization,
    # )

    SSV2_ROOT = Path("../dataset/ssv2")
    LABELS_DIR = SSV2_ROOT / "labels"  # contains train.json, validation.json, test.json, labels

    with open(LABELS_DIR / "labels.json") as f:
        class_to_idx = json.load(f)
    labeled_video_paths = make_dataset(
        SSV2_ROOT / "20bn-something-something-v2",
        class_to_idx,
        ".mp4"
    )
    clip_sampler = UniformClipSampler(clip_duration=3.0)

    dset = LabeledVideoDataset(
        video_sampler=torch.utils.data.SequentialSampler,
        labeled_video_paths=labeled_video_paths,
        clip_sampler=clip_sampler,
        transform=preprocessing_without_normalization,
    )

    indices = np.random.choice(len(labeled_video_paths), size=SUBSET_NUM_SAMPLES, replace=False)
    subset_paths = [labeled_video_paths[i] for i in indices]

    dset = SizedLabeledVideoDataset(
        labeled_video_paths=subset_paths,
        clip_sampler=clip_sampler,
        transform=preprocessing_without_normalization,
    )
    print(len(dset))
    text_to_concept.train_linear_aligner(dset, batch_size=16, load_reps=False, save_dir='data/representations_20k')
    text_to_concept.save_linear_aligner('videomae_ssv2_aligner_20k.pth')


if __name__ == '__main__':
    main()
