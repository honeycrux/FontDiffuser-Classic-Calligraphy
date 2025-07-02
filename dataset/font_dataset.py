# This script is provided by authors of FontDiffuser.
# This script defines the dataset for training of FontDiffuser.

from pathlib import Path
import random
from PIL import Image
from collections import defaultdict
import hashlib

import torch
from torch.utils.data import Dataset

from utils import get_transform_function

image_suffix = "png"


def parse_target_image_name(target_image_name: str):
    # Input Format: style+content[+optional-suffix]
    target_components = target_image_name.split("+")
    style = target_components[0]
    content = target_components[1]
    return style, content


def is_for_validation(image_char: str, validation_factor: int = 10):
    # Using the filename of a data, determine whether it is for validation
    # Overall, (1 / validation_factor) of the data is determined as validation data

    hash_value = int(hashlib.md5(image_char.encode()).hexdigest(), 16)
    is_validation = hash_value % validation_factor == 0
    return is_validation


class FontDataset(Dataset):
    """The dataset of font generation"""

    def __init__(
        self,
        args,
        phase: str,
        transforms,
        is_validation_mode: bool,
    ):
        super().__init__()
        self.root = args.data_root
        self.phase = phase
        self.use_scr = bool(args.use_scr)
        self.use_validation = args.use_validation
        self.validation_factor = args.validation_factor
        self.is_validation_mode = is_validation_mode
        if self.use_scr:
            self.num_neg = args.num_neg
        if self.is_validation_mode and not self.use_validation:
            raise ValueError(
                "User does not want to split validation set, but is in validation mode"
            )

        # Get Data path
        self.get_path()
        self.transforms = transforms
        self.nonorm_transforms = get_transform_function(
            target_size=(args.resolution, args.resolution), normalize=False
        )

    def get_path(self):
        # Find target image list style to images map
        self.target_images: list[str] = []
        self.content_to_images: dict[str, str] = {}
        self.style_to_images: dict[str, defaultdict[str, list[str]]] = {}
        content_image_dir = Path(self.root) / self.phase / "ContentImage"
        target_image_dir = Path(self.root) / self.phase / "TargetImage"
        message_prefix = (
            f"In {'validation' if self.is_validation_mode else 'training'} set:"
        )
        for style in target_image_dir.iterdir():
            if not style.is_dir():
                continue
            style_related_images = defaultdict[str, list[str]](list)
            for img in style.iterdir():
                image_style, image_char = parse_target_image_name(img.stem)
                if (
                    self.use_validation
                    and self.is_validation_mode
                    != is_for_validation(
                        image_char=image_char, validation_factor=self.validation_factor
                    )
                ):
                    continue
                img_path = img.as_posix()
                if image_char not in self.content_to_images:
                    content_path = content_image_dir / f"{image_char}.{image_suffix}"
                    assert (
                        content_path.exists()
                    ), f"{message_prefix} Content image {image_char} required by style {image_style} not found in {content_path}"
                    self.content_to_images[image_char] = content_path.as_posix()
                assert (
                    style.stem == image_style
                ), f"{message_prefix} Style mismatch: Expected {style.stem}, but got {image_style} in {img_path}"
                assert (
                    image_suffix == img.suffix[1:]
                ), f"{message_prefix} Image suffix mismatch: Expected {image_suffix}, but got {img.suffix} in {img_path}"
                self.target_images.append(img_path)
                style_related_images[image_char].append(img_path)
            self.style_to_images[style.stem] = style_related_images

        # Check the number of style images available at every situation
        required_style_images = 1
        for style, char_images_map in self.style_to_images.items():
            style_images_total = sum(
                [len(imlist) for imlist in char_images_map.values()]
            )
            for char, images in char_images_map.items():
                style_candidates_total = style_images_total - len(images)
                assert (
                    style_candidates_total >= required_style_images
                ), f"{message_prefix} When simulating training with style {style} and char {char}, the number of style images should be at least {required_style_images}, but got {style_candidates_total} style image candidates."

        # SCR: Check the number of styles
        num_styles = len(self.style_to_images)
        if self.use_scr:
            assert (
                num_styles >= self.num_neg + 1
            ), f"{message_prefix} To use SCR, the number of styles in TargetImage should be at least num_neg + 1, but got {num_styles} styles and {self.num_neg} num_neg."

        # SCR: Check if dataset is balanced (all styles have the same set of characters)
        if self.use_scr:
            universe_char_set = set(self.content_to_images.keys())
            empty_set = set()
            for style, char_images_map in self.style_to_images.items():
                style_char_set = set(char_images_map.keys())
                missing_set = universe_char_set - style_char_set
                if missing_set != empty_set:
                    raise Exception(
                        f"{message_prefix} When using SCR, a balance dataset is required (all styles should have the same set of characters), but got missing characters {missing_set} in style {style}."
                    )

    def __getitem__(self, index):
        target_image_path = Path(self.target_images[index])
        target_image_name = target_image_path.stem

        # Get target image components
        style, content = parse_target_image_name(target_image_name)

        # Read content image
        content_image_path = self.content_to_images[content]
        content_image = Image.open(content_image_path).convert("RGB")

        # Random sample used for style image
        char_images_map = self.style_to_images[style].copy()
        char_images_map.pop(content)
        candidate_style_images = [
            im for imlist in char_images_map.values() for im in imlist
        ]

        style_image_path = random.choice(candidate_style_images)
        style_image = Image.open(style_image_path).convert("RGB")

        # Read target image
        target_image = Image.open(target_image_path).convert("RGB")
        nonorm_target_image = self.nonorm_transforms(target_image)

        content_image = self.transforms[0](content_image)
        style_image = self.transforms[1](style_image)
        target_image = self.transforms[2](target_image)

        sample = {
            "content_image": content_image,
            "style_image": style_image,
            "target_image": target_image,
            "target_image_path": target_image_path.as_posix(),
            "nonorm_target_image": nonorm_target_image,
        }

        if self.use_scr:
            # Get neg image from the different style of the same content
            style_list = list(self.style_to_images.keys())
            style_list.remove(style)
            chosen_neg_paths = []
            chosen_styles = random.sample(style_list, self.num_neg)
            chosen_neg_paths = [
                random.choice(self.style_to_images[chosen_style][content])
                for chosen_style in chosen_styles
            ]

            # Load neg_images
            neg_images = None
            for i, neg_path in enumerate(chosen_neg_paths):
                neg_image = Image.open(neg_path).convert("RGB")
                neg_image = self.transforms[2](neg_image)
                assert isinstance(neg_image, torch.Tensor)
                if i == 0:
                    neg_images = neg_image[None, :, :, :]
                else:
                    assert neg_images is not None
                    neg_images = torch.cat(
                        [neg_images, neg_image[None, :, :, :]], dim=0
                    )
            assert neg_images is not None
            sample["neg_images"] = neg_images

        return sample

    def __len__(self):
        return len(self.target_images)
