import os
import random
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

def get_nonorm_transform(resolution):
    nonorm_transform =  transforms.Compose(
            [transforms.Resize((resolution, resolution), 
                               interpolation=transforms.InterpolationMode.BILINEAR), 
             transforms.ToTensor()])
    return nonorm_transform


class FontDataset(Dataset):
    """The dataset of font generation  
    """
    def __init__(self, args, phase, training_phase, transforms=None):
        super().__init__()
        self.root = args.data_root
        self.phase = phase
        self.training_phase = training_phase
        self.scr = training_phase >= 2
        self.k_shot = args.k_shot
        if self.scr:
            self.num_neg = args.num_neg
        
        # Get Data path
        self.get_path()
        self.transforms = transforms
        self.nonorm_transforms = get_nonorm_transform(args.resolution)

    def get_path(self):
        self.target_images = []
        # images with related style  
        self.style_to_images = {}
        target_image_dir = f"{self.root}/{self.phase}/TargetImage"
        for style in os.listdir(target_image_dir):
            images_related_style = []
            for img in os.listdir(f"{target_image_dir}/{style}"):
                img_path = f"{target_image_dir}/{style}/{img}"
                self.target_images.append(img_path)
                images_related_style.append(img_path)
            self.style_to_images[style] = images_related_style

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
            raise ValueError(f"k_shot is set to {self.k_shot}, but the number of style images is less than {self.k_shot}")
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
                choose_style = random.choice(style_list)
                choose_index = style_list.index(choose_style)
                style_list.pop(choose_index)
                choose_neg_name = f"{self.root}/train/TargetImage/{choose_style}/{choose_style}+{content}.png"
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
