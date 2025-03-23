import os
import random
from PIL import Image
from collections import defaultdict

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

def get_nonorm_transform(resolution):
    nonorm_transform =  transforms.Compose(
            [transforms.Resize((resolution, resolution), 
                               interpolation=transforms.InterpolationMode.BILINEAR), 
             transforms.ToTensor()])
    return nonorm_transform

def parse_target_image_name(target_image_name: str):
    # Input Format: style+content[+optional-suffix].png
    target_components = target_image_name.split('.')[0].split('+')
    style = target_components[0]
    content = target_components[1]
    return style, content


class FontDataset(Dataset):
    """The dataset of font generation  
    """
    def __init__(self, args, phase, transforms, scr):
        super().__init__()
        self.root = args.data_root
        self.phase = phase
        self.scr = bool(scr)
        if self.scr:
            self.num_neg = args.num_neg
        
        # Get Data path
        self.get_path()
        self.transforms = transforms
        self.nonorm_transforms = get_nonorm_transform(args.resolution)

    def get_path(self):
        self.target_images: list[str] = []
        # images with related style  
        self.style_to_images: dict[str, defaultdict[str, list[str]]] = {}
        target_image_dir = f"{self.root}/{self.phase}/TargetImage"
        for style in os.listdir(target_image_dir):
            style_related_images = defaultdict[str, list[str]](list)
            for img in os.listdir(f"{target_image_dir}/{style}"):
                image_style, image_char = parse_target_image_name(img)
                img_path = f"{target_image_dir}/{style}/{img}"
                assert image_style == style, f"Style mismatch: {style} vs {image_style} in {img_path}"
                self.target_images.append(img_path)
                style_related_images[image_char].append(img_path)
            self.style_to_images[style] = style_related_images

    def __getitem__(self, index):
        target_image_path = self.target_images[index]
        target_image_name = target_image_path.split('/')[-1]

        # Get target image components
        style, content = parse_target_image_name(target_image_name)
        
        # Read content image
        content_image_path = f"{self.root}/{self.phase}/ContentImage/{content}.png"
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
            "target_image_path": target_image_path,
            "nonorm_target_image": nonorm_target_image}
        
        if self.scr:
            # Get neg image from the different style of the same content
            style_list = list(self.style_to_images.keys())
            style_index = style_list.index(style)
            style_list.pop(style_index)
            choose_neg_names = []
            for i in range(self.num_neg):
                choose_style = random.choice(style_list)
                choose_index = style_list.index(choose_style)
                style_list.pop(choose_index)
                choose_neg_name = f"{self.root}/train/TargetImage/{choose_style}/{choose_style}+{content}.png"
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
