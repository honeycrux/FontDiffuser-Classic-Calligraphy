from pathlib import Path
import random
from PIL import Image
import hashlib

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

def get_nonorm_transform(resolution):
    nonorm_transform =  transforms.Compose(
            [transforms.Resize((resolution, resolution), 
                               interpolation=transforms.InterpolationMode.BILINEAR), 
             transforms.ToTensor()])
    return nonorm_transform

def is_for_validation(filename):
    # Using the filename of a data, determine whether it is for validation
    # Overall, 10% of the data should be determined as validation data

    hash_value = int(hashlib.md5(filename.encode()).hexdigest(), 16)
    is_validation = hash_value % 10 == 0
    return is_validation

class FontDataset(Dataset):
    """The dataset of font generation  
    """
    def __init__(self, args, phase, scr, need_validation_split, is_validation_mode, transforms=None):
        super().__init__()
        self.root = args.data_root
        self.phase = phase
        self.validate_set_size_limit = args.validate_set_size
        self.need_validation_split = bool(need_validation_split)
        self.is_validation_mode = bool(is_validation_mode)
        self.scr = bool(scr)
        self.k_shot = args.k_shot
        if self.scr:
            self.num_neg = args.num_neg
        if self.is_validation_mode and not self.need_validation_split:
            raise ValueError("User does not want to split validation set, but is in validation mode")

        # Get Data path
        self.get_path()
        self.transforms = transforms
        self.nonorm_transforms = get_nonorm_transform(args.resolution)

    def get_path(self):

        self.target_images = []
        # Images with related style
        self.style_to_images = {}
        target_image_dir = Path(self.root) / self.phase / "TargetImage"
        number_of_styles = len(list(target_image_dir.iterdir()))
        # Limit the number of images per style so that the size of the whole validation set is at most validate_set_size_limit
        print(f"Number of styles in dataset: {number_of_styles}")
        if self.is_validation_mode and self.need_validation_split:
            limit_per_style = (self.validate_set_size_limit // number_of_styles) if self.validate_set_size_limit is not None else None
            if limit_per_style:
                print(f"Validation mode with limit: {self.validate_set_size_limit}, Limit per style: {self.validate_set_size_limit}//{number_of_styles} = {limit_per_style}")
                if limit_per_style < self.k_shot + 1:
                    raise ValueError(f"limit_per_style is set to {limit_per_style}, but it should be at least {self.k_shot} + 1 \n"
                                     f"Did you mean to set validate_set_size to at least {(self.k_shot + 1) * number_of_styles}?")
            else:
                print("Validation mode with limit: Unlimited (some percentage of the data determined by is_for_validation, usually 10%)")
        for style in target_image_dir.iterdir():
            images_related_style = []
            for idx, img in enumerate(style.iterdir()):
                if self.is_validation_mode and limit_per_style is not None and len(images_related_style) >= limit_per_style:
                    break
                if self.need_validation_split and self.is_validation_mode != is_for_validation(img.stem):
                    continue
                img_path = img.as_posix()
                self.target_images.append(img_path)
                images_related_style.append(img_path)
            self.style_to_images[style.name] = images_related_style

    def __getitem__(self, index):
        target_image_path = self.target_images[index]
        target_image_name = target_image_path.split('/')[-1]
        style, content = target_image_name.split('.')[0].split('+')
        
        # Read content image
        content_image_path = f"{self.root}/{self.phase}/ContentImage/{content}.png"
        content_image = Image.open(content_image_path).convert('RGB')
        if self.transforms is not None:
            content_image = self.transforms[0](content_image)

        # Random sample used for style image
        images_related_style = self.style_to_images[style].copy()
        images_related_style.remove(target_image_path)

        # Original implementation: Get 1 style image
        # style_image_path = random.choice(images_related_style)
        # style_image = Image.open(style_image_path).convert("RGB")
        # if self.transforms is not None:
        #     style_image = self.transforms[1](style_image)

        # My implementation: Get K style images of the same style
        choose_style_image_names = []
        # Choose style images
        if len(images_related_style) < self.k_shot:
            raise ValueError(f"k_shot is set to {self.k_shot}, but the number of style images ({len(images_related_style)}) is less than {self.k_shot}")
        for i in range(self.k_shot):
            style_image_path = random.choice(images_related_style)
            choose_style_image_names.append(style_image_path)
            images_related_style.remove(style_image_path)
        # Load style images
        for i, style_image_path in enumerate(choose_style_image_names):
            style_image = Image.open(style_image_path).convert("RGB")
            if self.transforms is not None:
                style_image = self.transforms[1](style_image)
            if i == 0:
                style_images = style_image[None, :, :, :]
            else:
                style_images = torch.cat([style_images, style_image[None, :, :, :]], dim=0)
        
        # Read target image
        target_image = Image.open(target_image_path).convert("RGB")
        nonorm_target_image = self.nonorm_transforms(target_image)
        if self.transforms is not None:
            target_image = self.transforms[2](target_image)
        
        sample = {
            "content_image": content_image,
            "style_images": style_images,
            "target_image": target_image,
            "target_image_path": target_image_path,
            "nonorm_target_image": nonorm_target_image}
        
        if self.scr:
            # Get neg image from the different style of the same content
            style_list = list(self.style_to_images.keys())
            style_index = style_list.index(style)
            style_list.pop(style_index)
            choose_neg_names = []
            for i in range(self.num_neg):
                if len(style_list) < 1:
                    # choose less than num_neg if there is not enough other styles
                    break
                choose_style = random.choice(style_list)
                choose_index = style_list.index(choose_style)
                style_list.pop(choose_index)
                choose_neg_name = f"{self.root}/{self.phase}/TargetImage/{choose_style}/{choose_style}+{content}.png"
                choose_neg_names.append(choose_neg_name)

            # Load neg_images
            for i, neg_name in enumerate(choose_neg_names):
                neg_image = Image.open(neg_name).convert("RGB")
                if self.transforms is not None:
                    neg_image = self.transforms[2](neg_image)
                if i == 0:
                    neg_images = neg_image[None, :, :, :]
                else:
                    neg_images = torch.cat([neg_images, neg_image[None, :, :, :]], dim=0)
            sample["neg_images"] = neg_images

        return sample

    def __len__(self):
        return len(self.target_images)
