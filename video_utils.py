import os
import torch
import random
from pathlib import Path

from torchvision.datasets.folder import has_file_allowed_extension
from torchvision import transforms, datasets
from typing import Callable, Optional, Union, cast

IMAGENET_MEAN = (0.485, 0.456, 0.406)  # TODO review
IMAGENET_STD = (0.229, 0.224, 0.225)


def make_dataset(
    directory: Union[str, Path],
    class_to_idx: Optional[dict[str, int]] = None,
    extensions: Optional[Union[str, tuple[str, ...]]] = None,
    is_valid_file: Optional[Callable[[str], bool]] = None,
    allow_empty: bool = False,
) -> list[tuple[str, int]]:
    """
    Note: Function adapted from torchvision for a simpler dataset structure.

    Generates a list of samples of a form (path_to_sample, class).

    See :class:`DatasetFolder` for details.

    Note: The class_to_idx parameter is here optional and will use the logic of the ``find_classes`` function
    by default.
    """
    directory = os.path.expanduser(directory)

    if not class_to_idx:
        raise ValueError("'class_to_index' must have at least one entry to collect any samples.")

    both_none = extensions is None and is_valid_file is None
    both_something = extensions is not None and is_valid_file is not None
    if both_none or both_something:
        raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")

    if extensions is not None:

        def is_valid_file(x: str) -> bool:
            return has_file_allowed_extension(x, extensions)  # type: ignore[arg-type]

    is_valid_file = cast(Callable[[str], bool], is_valid_file)

    instances = []
    available_classes = set()
    for target_class in sorted(class_to_idx.keys()):
        class_index = {"label": class_to_idx[target_class]}
        target_dir = directory #os.path.join(directory, target_class)
        if not os.path.isdir(target_dir):
            continue
        for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                if is_valid_file(path):
                    item = path, class_index
                    instances.append(item)

                    if target_class not in available_classes:
                        available_classes.add(target_class)

    empty_classes = set(class_to_idx.keys()) - available_classes
    if empty_classes and not allow_empty:
        msg = f"Found no valid file for the classes {', '.join(sorted(empty_classes))}. "
        if extensions is not None:
            msg += f"Supported extensions are: {extensions if isinstance(extensions, str) else ', '.join(extensions)}"
        raise FileNotFoundError(msg)

    return instances


class VideoMAETTCTWrapper(torch.nn.Module):
    def __init__(self, model, normalizer=None, mtype="videomae"):
        super().__init__()
        self.mtype = mtype
        self.model = model
        self.normalizer = normalizer

    def forward_features(self, x):
        sequence_videomae_feats = self.model.videomae(
            pixel_values=x
        ).last_hidden_state

        if self.model.fc_norm is not None:
            videomae_feats = sequence_videomae_feats.mean(1)
            videomae_feats = self.model.fc_norm(videomae_feats)
        else:
            videomae_feats = sequence_videomae_feats[:, 0]

        return videomae_feats

    def get_normalizer(self, x):
        if self.normalizer is None:
            return x
        return self.normalizer(x)

    @property
    def has_normalizer(self):
        return self.normalizer is not None

class ToTensorTuple:
    def __init__(self, key_list):
        self.key_list = key_list

    def __call__(self, x, *args, **kwds):
        return tuple(x[key] for key in self.key_list)  # TODO see batching


class DivideBy255:
    def __call__(self, x):
        return x / 255.0


class CTHWToTCHW:
    def __call__(self, x):
        return x.permute(1, 0, 2, 3)
