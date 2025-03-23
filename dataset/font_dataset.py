from pathlib import Path
import random
from PIL import Image
from collections import defaultdict
import hashlib

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

image_suffix = "png"

def get_nonorm_transform(resolution):
    nonorm_transform =  transforms.Compose(
            [transforms.Resize((resolution, resolution), 
                               interpolation=transforms.InterpolationMode.BILINEAR), 
             transforms.ToTensor()])
    return nonorm_transform

def parse_target_image_name(target_image_name: str):
    # Input Format: style+content[+optional-suffix]
    target_components = target_image_name.split('+')
    style = target_components[0]
    content = target_components[1]
    return style, content

def is_for_validation(filename: str, validation_factor: int = 10):
    # Using the filename of a data, determine whether it is for validation
    # Overall, (1 / validation_factor) of the data is determined as validation data

    hash_value = int(hashlib.md5(filename.encode()).hexdigest(), 16)
    is_validation = hash_value % validation_factor == 0
    return is_validation

class FontDataset(Dataset):
    """The dataset of font generation  
    """
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
            raise ValueError("User does not want to split validation set, but is in validation mode")
        
        # Get Data path
        self.get_path()
        self.transforms = transforms
        self.nonorm_transforms = get_nonorm_transform(args.resolution)

    def get_path(self):
        self.target_images: list[str] = []
        # images with related style  
        self.style_to_images: dict[str, defaultdict[str, list[str]]] = {}
        target_image_dir = Path(self.root) / self.phase / "TargetImage"
        for style in target_image_dir.iterdir():
            if not style.is_dir():
                continue
            style_related_images = defaultdict[str, list[str]](list)
            for img in style.iterdir():
                if self.use_validation and self.is_validation_mode != is_for_validation(img.stem, self.validation_factor):
                    continue
                image_style, image_char = parse_target_image_name(img.stem)
                img_path = img.as_posix()
                assert style.stem == image_style, f"Style mismatch: Expected {style.stem}, but got {image_style} in {img_path}"
                assert image_suffix == img.suffix[1:], f"Image suffix mismatch: Expected {image_suffix}, but got {img.suffix} in {img_path}"
                self.target_images.append(img_path)
                style_related_images[image_char].append(img_path)
            self.style_to_images[style.stem] = style_related_images

    def __getitem__(self, index):
        target_image_path = Path(self.target_images[index])
        target_image_name = target_image_path.stem

        # Get target image components
        style, content = parse_target_image_name(target_image_name)
        
        # Read content image
        content_image_path = f"{self.root}/{self.phase}/ContentImage/{content}.{image_suffix}"
        content_image = Image.open(content_image_path).convert('RGB')

        # Random sample used for style image
        style_imlist_map = self.style_to_images[style].copy()
        style_imlist_map.pop(content)
        candidate_style_images = [im for imlist in style_imlist_map.values() for im in imlist]

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
            "nonorm_target_image": nonorm_target_image}
        
        if self.use_scr:
            # Get neg image from the different style of the same content
            style_list = list(self.style_to_images.keys())
            style_index = style_list.index(style)
            style_list.pop(style_index)
            choose_neg_names = []
            for i in range(self.num_neg):
                choose_style = random.choice(style_list)
                choose_index = style_list.index(choose_style)
                style_list.pop(choose_index)
                choose_neg_name = f"{self.root}/train/TargetImage/{choose_style}/{choose_style}+{content}.{image_suffix}"
                choose_neg_names.append(choose_neg_name)

            # Load neg_images
            neg_images = None
            for i, neg_name in enumerate(choose_neg_names):
                neg_image = Image.open(neg_name).convert("RGB")
                neg_image = self.transforms[2](neg_image)
                assert isinstance(neg_image, torch.Tensor)
                if i == 0:
                    neg_images = neg_image[None, :, :, :]
                else:
                    assert neg_images is not None
                    neg_images = torch.cat([neg_images, neg_image[None, :, :, :]], dim=0)
            assert neg_images is not None
            sample["neg_images"] = neg_images

        return sample

    def __len__(self):
        return len(self.target_images)
