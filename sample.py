# This script is provided by authors of FontDiffuser.
# This script is the sampling process of FontDiffuser.
# For usage, also refer to lantingjixu_sample.py or scripts/sample_content_*.sh.

import os
import random
import time
from pathlib import Path
from typing import Union

import cv2
import numpy as np
import torch
from accelerate.utils import set_seed
from PIL import Image

from src import (
    FontDiffuserDPMPipeline,
    FontDiffuserModelDPM,
    build_content_encoder,
    build_ddpm_scheduler,
    build_style_absorption,
    build_style_encoder,
    build_unet,
)
from utils import (
    get_transform_function,
    is_char_in_font,
    load_ttf,
    save_args_to_yaml,
    save_image_with_content_style,
    save_single_image,
    ttf2im,
)


class SourceImage:
    image: Image.Image

    def __init__(self, image):
        assert isinstance(image, Image.Image), "The image should be PIL.Image.Image."
        self.image = image

    @staticmethod
    def from_args(args) -> "SourceImage":
        if args.character_input:
            assert isinstance(
                args.content_character, str
            ), "The content_character should be provided when character_input is True."
            assert isinstance(
                args.ttf_path, str
            ), "The ttf_path should be provided when character_input is True."
            assert is_char_in_font(
                font_path=args.ttf_path, char=args.content_character
            ), "The content_character is not in the ttf. \
                    Please change the content_character or you can change the ttf."
            font = load_ttf(ttf_path=args.ttf_path)
            content_image = ttf2im(font=font, char=args.content_character)
        else:
            assert isinstance(
                args.content_image_path, str
            ), "The content_image_path should be str."
            content_image = Image.open(args.content_image_path).convert("RGB")

        return SourceImage(image=content_image)


class ReferenceImage:
    actual_image: Image.Image
    computer_font_image_or_character: Union[Image.Image, str]

    def __init__(
        self,
        actual_image,
        computer_font_image_or_character,
    ):
        assert isinstance(
            actual_image, Image.Image
        ), "The actual_image should be PIL.Image.Image."
        assert isinstance(
            computer_font_image_or_character, (Image.Image, str)
        ), "The computer_font_image_or_character should be PIL.Image.Image or str."
        self.actual_image = actual_image
        if isinstance(computer_font_image_or_character, str):
            assert (
                len(computer_font_image_or_character) == 1
            ), f"If computer_font_image_or_character is str, it should be a single character string, \
                but got {computer_font_image_or_character}."
        self.computer_font_image_or_character = computer_font_image_or_character

    def get_computer_font_image(
        self, computer_font_dir=None, ttf_path=None
    ) -> Image.Image:
        # Case 1: already provided computer font image
        if isinstance(self.computer_font_image_or_character, Image.Image):
            return self.computer_font_image_or_character

        character = self.computer_font_image_or_character

        # Case 2: computer font image found in provided directory
        if isinstance(computer_font_dir, str):
            computer_font_path = Path(computer_font_dir) / f"{character}.png"
            if computer_font_path.exists():
                computer_font_image = Image.open(computer_font_path).convert("RGB")
                return computer_font_image

        # Case 3: generate computer font image from ttf
        assert isinstance(
            ttf_path, str
        ), "The ttf_path should be provided when computer_font_image needs to be generated."
        assert is_char_in_font(
            font_path=ttf_path, char=character
        ), f"The character {character} is not in the ttf. \
                Please change the character or you can change the ttf."

        font = load_ttf(ttf_path=ttf_path)
        computer_font_image = ttf2im(font=font, char=character)
        assert isinstance(
            computer_font_image, Image.Image
        ), f"The computer font image generation for {character} failed."

        return computer_font_image

    @staticmethod
    def from_image_path(actual_image_path: Path) -> "ReferenceImage":
        # Image File Name Format: style+character[+optional-suffix] or character.png

        actual_image = Image.open(actual_image_path).convert("RGB")

        image_name = actual_image_path.stem
        if "+" in image_name:
            target_components = image_name.split("+")
            character = target_components[1]
        else:
            character = image_name

        return ReferenceImage(
            actual_image=actual_image,
            computer_font_image_or_character=character,
        )


class ReferenceImageList:
    images: list[ReferenceImage]

    def __init__(self, images: list[ReferenceImage], k_shot):
        assert all(
            [isinstance(image, ReferenceImage) for image in images]
        ), "All elements in images should be ReferenceImage."
        assert isinstance(k_shot, int), "k_shot should be an integer."
        self.images = random.sample(images, k=min([k_shot, len(images)]))

    @staticmethod
    def from_args(args) -> "ReferenceImageList":
        assert isinstance(
            args.style_image_path, str
        ), "The style_image_path should be str."
        style_images_dir = Path(args.style_image_path)
        available_style_paths: list[Path] = []
        for path in style_images_dir.iterdir():
            if path.is_file():
                available_style_paths.append(path)
        num_style_images = len(available_style_paths)
        # Sample k_shot style images here to save memory, since ReferenceImage loads all images into memory.
        assert isinstance(args.k_shot, int), "args.k_shot should be an integer."
        style_image_paths = random.sample(
            available_style_paths, k=min([args.k_shot, num_style_images])
        )
        style_images = [
            ReferenceImage.from_image_path(path) for path in style_image_paths
        ]
        return ReferenceImageList(images=style_images, k_shot=args.k_shot)


def arg_parse():
    from configs.fontdiffuser import get_parser

    parser = get_parser()
    parser.add_argument("--ckpt_dir", type=str, default=None)
    parser.add_argument("--demo", action="store_true")
    parser.add_argument(
        "--controlnet",
        type=bool,
        default=False,
        help="If in demo mode, the controlnet can be added.",
    )
    parser.add_argument("--character_input", action="store_true")
    parser.add_argument("--content_character", type=str, default=None)
    parser.add_argument("--content_image_path", type=str, default=None)
    parser.add_argument("--style_image_path", type=str, default=None)
    parser.add_argument("--computer_font_image_dir", type=str, default=None)
    parser.add_argument("--save_image", action="store_true")
    parser.add_argument(
        "--save_image_dir", type=str, default=None, help="The saving directory."
    )
    parser.add_argument(
        "--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--ttf_path", type=str, default="ttf/KaiXinSongA.ttf")
    args = parser.parse_args()
    style_image_size = args.style_image_size
    content_image_size = args.content_image_size
    args.style_image_size = (style_image_size, style_image_size)
    args.content_image_size = (content_image_size, content_image_size)

    return args


def image_process(
    args,
    content_image: Union[None, SourceImage] = None,
    style_images: Union[None, ReferenceImageList] = None,
) -> Union[
    None,
    tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Image.Image,
        list[Image.Image],
        list[Image.Image],
    ],
]:
    ## Get PIL images

    if content_image is None:
        content_image = SourceImage.from_args(args=args)
    if style_images is None:
        style_images = ReferenceImageList.from_args(args=args)

    content_image_pil = content_image.image
    style_images_pil = [style_image.actual_image for style_image in style_images.images]
    style_images_in_computer_font_pil = [
        style_image.get_computer_font_image(
            computer_font_dir=args.computer_font_image_dir, ttf_path=args.ttf_path
        )
        for style_image in style_images.images
    ]

    ## Transform images to tensors

    content_transforms = get_transform_function(
        target_size=args.content_image_size, normalize=True
    )
    style_transforms = get_transform_function(
        target_size=args.style_image_size, normalize=True
    )

    # Apply the transform to the content image
    transformed_content_image = content_transforms(content_image_pil)[None, :]
    # Apply the transform to the style image
    transformed_style_images = [
        style_transforms(style_image)[None, :] for style_image in style_images_pil
    ]
    # Combine the style images into a single tensor
    transformed_style_images = torch.cat(transformed_style_images, dim=0)
    # Apply the transform to the style image in computer font
    transformed_style_images_in_computer_font = [
        style_transforms(style_image_in_computer_font)[None, :]
        for style_image_in_computer_font in style_images_in_computer_font_pil
    ]
    # Combine the style images in computer font into a single tensor
    transformed_style_images_in_computer_font = torch.cat(
        transformed_style_images_in_computer_font, dim=0
    )

    return (
        transformed_content_image,
        transformed_style_images,
        transformed_style_images_in_computer_font,
        content_image_pil,
        style_images_pil,
        style_images_in_computer_font_pil,
    )


def load_fontdiffuser_pipeline(args):
    # Load the model state_dict
    unet = build_unet(args=args)
    unet.load_state_dict(torch.load(f"{args.ckpt_dir}/unet.pth"))
    style_encoder = build_style_encoder(args=args)
    style_encoder.load_state_dict(torch.load(f"{args.ckpt_dir}/style_encoder.pth"))
    content_encoder = build_content_encoder(args=args)
    content_encoder.load_state_dict(torch.load(f"{args.ckpt_dir}/content_encoder.pth"))
    style_absorption = build_style_absorption(args=args)
    style_absorption.load_state_dict(
        torch.load(f"{args.ckpt_dir}/style_absorption.pth")
    )
    model = FontDiffuserModelDPM(
        unet=unet,
        style_encoder=style_encoder,
        content_encoder=content_encoder,
        style_absorption=style_absorption,
    )
    model.to(args.device)
    print("Loaded the model state_dict successfully!")

    # Load the training ddpm_scheduler.
    train_scheduler = build_ddpm_scheduler(args=args)
    print("Loaded training DDPM scheduler sucessfully!")

    # Load the DPM_Solver to generate the sample.
    pipe = FontDiffuserDPMPipeline(
        model=model,
        ddpm_train_scheduler=train_scheduler,
        model_type=args.model_type,
        guidance_type=args.guidance_type,
        guidance_scale=args.guidance_scale,
    )
    print("Loaded dpm_solver pipeline sucessfully!")

    return pipe


def sampling(args, pipe, content_image=None, style_images=None):
    if args.save_image:
        os.makedirs(args.save_image_dir, exist_ok=True)

        # saving sampling config
        save_args_to_yaml(
            args=args, output_file=f"{args.save_image_dir}/sampling_config.yaml"
        )

    if type(args.seed) is int:
        set_seed(seed=args.seed)

    image_process_output = image_process(
        args=args, content_image=content_image, style_images=style_images
    )

    if image_process_output == None:
        print(
            f"The content_character you provided is not in the ttf. \
                Please change the content_character or you can change the ttf."
        )
        return None

    (
        content_image,
        style_images,
        style_images_in_computer_font,
        content_image_pil,
        _,
        _,
    ) = image_process_output

    with torch.no_grad():
        content_image = content_image.to(args.device)
        style_images = style_images.to(args.device)
        style_images_in_computer_font = style_images_in_computer_font.to(args.device)
        print(f"Sampling by DPM-Solver++ ......")
        start = time.time()
        images = pipe.generate(
            content_images=content_image,
            style_images=style_images,
            style_images_in_computer_font=style_images_in_computer_font,
            batch_size=1,
            order=args.order,
            num_inference_step=args.num_inference_steps,
            content_encoder_downsample_size=args.content_encoder_downsample_size,
            t_start=args.t_start,
            t_end=args.t_end,
            dm_size=args.content_image_size,
            algorithm_type=args.algorithm_type,
            skip_type=args.skip_type,
            method=args.method,
            correcting_x0_fn=args.correcting_x0_fn,
        )
        end = time.time()

        if args.save_image:
            print(f"Saving the image ......")
            save_single_image(
                save_dir=args.save_image_dir,
                image=images[0],
                character=args.content_character,
            )
            save_image_with_content_style(
                save_dir=args.save_image_dir,
                image=images[0],
                character=args.content_character,
                content_image_pil=content_image_pil,
                content_image_path=None,
                style_image_path=args.style_image_path,
                resolution=args.resolution,
            )
            print(f"Finish the sampling process, costing time {end - start}s")
        return images[0]


def load_controlnet_pipeline(
    args,
    config_path="lllyasviel/sd-controlnet-canny",
    ckpt_path="runwayml/stable-diffusion-v1-5",
):
    from diffusers.models.controlnet import ControlNetModel

    # from diffusers.models.autoencoder_kl import AutoencoderKL
    # load controlnet model and pipeline
    from diffusers.pipelines.controlnet.pipeline_controlnet import (
        StableDiffusionControlNetPipeline,
    )
    from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler

    controlnet = ControlNetModel.from_pretrained(
        config_path, torch_dtype=torch.float16, cache_dir=f"{args.ckpt_dir}/controlnet"
    )
    print(f"Loaded ControlNet Model Successfully!")
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        ckpt_path,
        controlnet=controlnet,
        torch_dtype=torch.float16,
        cache_dir=f"{args.ckpt_dir}/controlnet_pipeline",
    )
    # faster
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()
    print(f"Loaded ControlNet Pipeline Successfully!")

    return pipe


def controlnet(text_prompt, pil_image, pipe):
    image = np.array(pil_image)
    # get canny image
    image = cv2.Canny(image=image, threshold1=100, threshold2=200)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    canny_image = Image.fromarray(image)

    seed = random.randint(0, 10000)
    generator = torch.manual_seed(seed)
    image = pipe(
        text_prompt,
        num_inference_steps=50,
        generator=generator,
        image=canny_image,
        output_type="pil",
    ).images[0]
    return image


def load_instructpix2pix_pipeline(args, ckpt_path="timbrooks/instruct-pix2pix"):
    from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_instruct_pix2pix import (
        StableDiffusionInstructPix2PixPipeline,
    )
    from diffusers.schedulers.scheduling_euler_ancestral_discrete import (
        EulerAncestralDiscreteScheduler,
    )

    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        ckpt_path, torch_dtype=torch.float16
    )
    pipe.to(args.device)
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

    return pipe


def instructpix2pix(pil_image, text_prompt, pipe):
    image = pil_image.resize((512, 512))
    seed = random.randint(0, 10000)
    generator = torch.manual_seed(seed)
    image = pipe(
        prompt=text_prompt,
        image=image,
        generator=generator,
        num_inference_steps=20,
        image_guidance_scale=1.1,
    ).images[0]

    return image


if __name__ == "__main__":
    args = arg_parse()

    # load fontdiffuser pipeline
    pipe = load_fontdiffuser_pipeline(args=args)
    out_image = sampling(args=args, pipe=pipe)
