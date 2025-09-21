# This script is provided by authors of FontDiffuser.
# This script contains utility functions used in the FontDiffuser scripts.

import copy

import cv2
import numpy as np
import pygame
import pygame.freetype
import torch
import torchvision.transforms as transforms
import yaml
from fontTools.ttLib import TTFont
from PIL import Image


def save_args_to_yaml(args, output_file):
    # Convert args namespace to a dictionary
    args_dict = vars(args)

    # Write the dictionary to a YAML file
    with open(output_file, "w") as yaml_file:
        yaml.dump(args_dict, yaml_file, default_flow_style=False)


def get_file_name(character):
    assert (
        isinstance(character, str) or character is None
    ), "Character must be a string or None"

    if character is None:
        return "out"
    if character.isspace():
        return "empty"
    return character


def save_single_image(save_dir, image, character):
    file_name = get_file_name(character)
    save_path = f"{save_dir}/{file_name}.png"
    image.save(save_path)


def save_image_with_content_style(
    save_dir,
    image,
    character,
    content_image_pil,
    content_image_path,
    style_image_path,
    resolution,
):

    new_image = Image.new("RGB", (resolution * 3, resolution))
    if content_image_pil is not None:
        content_image = content_image_pil
    else:
        content_image = (
            Image.open(content_image_path)
            .convert("RGB")
            .resize((resolution, resolution), Image.Resampling.BILINEAR)
        )
    style_image = (
        Image.open(style_image_path)
        .convert("RGB")
        .resize((resolution, resolution), Image.Resampling.BILINEAR)
    )

    new_image.paste(content_image, (0, 0))
    new_image.paste(style_image, (resolution, 0))
    new_image.paste(image, (resolution * 2, 0))

    file_name = get_file_name(character)
    save_path = f"{save_dir}/{file_name}_with_cs.png"
    new_image.save(save_path)


def x0_from_epsilon(scheduler, noise_pred, x_t, timesteps):
    """Return the x_0 from epsilon"""
    batch_size = noise_pred.shape[0]
    pred_original_sample = None
    for i in range(batch_size):
        noise_pred_i = noise_pred[i]
        noise_pred_i = noise_pred_i[None, :]
        t = timesteps[i]
        x_t_i = x_t[i]
        x_t_i = x_t_i[None, :]

        pred_original_sample_i = scheduler.step(
            model_output=noise_pred_i,
            timestep=t,
            sample=x_t_i,
            # predict_epsilon=True,
            generator=None,
            return_dict=True,
        ).pred_original_sample
        if i == 0:
            pred_original_sample = pred_original_sample_i
        else:
            assert pred_original_sample is not None
            pred_original_sample = torch.cat(
                (pred_original_sample, pred_original_sample_i), dim=0
            )

    assert pred_original_sample is not None
    return pred_original_sample


def reNormalize_img(pred_original_sample):
    pred_original_sample = (pred_original_sample / 2 + 0.5).clamp(0, 1)

    return pred_original_sample


def normalize_mean_std(image):
    transforms_norm = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    image = transforms_norm(image)

    return image


def is_char_in_font(font_path, char):
    TTFont_font = TTFont(font_path)
    cmap = TTFont_font["cmap"]
    for subtable in cmap.tables:
        if ord(char) in subtable.cmap:
            return True
    return False


def load_ttf(ttf_path, fsize=128):
    pygame.init()

    font = pygame.freetype.Font(ttf_path, size=fsize)
    return font


def ttf2im(font, char, fsize=128):

    try:
        surface, _ = font.render(char)
    except:
        print("No glyph for char {}".format(char))
        return
    bg = np.full((fsize, fsize), 255)
    imo = pygame.surfarray.pixels_alpha(surface).transpose(1, 0)
    imo = 255 - np.array(Image.fromarray(imo))
    im = copy.deepcopy(bg)
    h, w = imo.shape[:2]
    if h > fsize:
        h, w = fsize, round(w * fsize / h)
        imo = cv2.resize(imo, (w, h))
    if w > fsize:
        h, w = round(h * fsize / w), fsize
        imo = cv2.resize(imo, (w, h))
    x, y = round((fsize - w) / 2), round((fsize - h) / 2)
    im[y : h + y, x : x + w] = imo
    pil_im = Image.fromarray(im.astype("uint8")).convert("RGB")

    return pil_im


def get_transform_function(target_size: tuple[int, int], normalize: bool):
    """Get a transform function for transforming the input image to a tensor with the specified resolution.
    This pads the image to make it square and resizes it to the resolution specified.
    """

    def apply_transform(img: Image.Image):
        width, height = img.size

        max_dim = max(width, height)
        padding = (
            (max_dim - width) // 2,  # left
            (max_dim - height) // 2,  # top
            (max_dim - width + 1) // 2,  # right
            (max_dim - height + 1) // 2,  # bottom
        )

        transforms_list = [
            transforms.Pad(padding=padding, fill=255, padding_mode="constant"),
            transforms.Resize(
                target_size, interpolation=transforms.InterpolationMode.BILINEAR
            ),
            transforms.ToTensor(),
        ]

        if normalize:
            transforms_list.append(transforms.Normalize([0.5], [0.5]))

        image_transforms = transforms.Compose(transforms_list)

        transformed_image = image_transforms(img)
        assert isinstance(transformed_image, torch.Tensor)
        return transformed_image

    return apply_transform
