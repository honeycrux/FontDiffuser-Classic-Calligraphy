# This script is provided by authors of FontDiffuser.
# This script is the sampling process of FontDiffuser.
# For usage, also refer to lantingjixu_sample.py or scripts/sample_content_*.sh.

import os
import random
import time
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


def image_process_with_path(args) -> Union[None, tuple[Image.Image, Image.Image]]:
    content_image_path = args.content_image_path
    style_image_path = args.style_image_path
    content_character = args.content_character

    if args.character_input:
        assert isinstance(
            content_character, str
        ), "The content_character should be str."
        if not (
            args.ttf_path
            and is_char_in_font(font_path=args.ttf_path, char=content_character)
        ):
            return None
        font = load_ttf(ttf_path=args.ttf_path)
        content_image = ttf2im(font=font, char=content_character)
    else:
        assert isinstance(
            content_image_path, str
        ), "The content_image_path should be str."
        content_image = Image.open(content_image_path).convert("RGB")

    assert isinstance(style_image_path, str), "The style_image_path should be str."
    style_image = Image.open(style_image_path).convert("RGB")

    assert isinstance(
        content_image, Image.Image
    ), "The content image should be PIL.Image.Image."
    assert isinstance(
        style_image, Image.Image
    ), "The style image should be PIL.Image.Image."

    return content_image, style_image


def image_process_with_image(
    args, content_image, style_image
) -> Union[None, tuple[Image.Image, Image.Image]]:
    content_character = args.content_character

    if args.character_input:
        assert isinstance(
            content_character, str
        ), "The content_character should be str."
        if not (
            args.ttf_path
            and is_char_in_font(font_path=args.ttf_path, char=content_character)
        ):
            return None
        font = load_ttf(ttf_path=args.ttf_path)
        content_image = ttf2im(font=font, char=args.content_character)

    assert isinstance(
        content_image, Image.Image
    ), "The content image should be PIL.Image.Image."
    assert isinstance(
        style_image, Image.Image
    ), "The style image should be PIL.Image.Image."

    return content_image, style_image


def image_process(
    args, content_image=None, style_image=None
) -> Union[None, tuple[torch.Tensor, torch.Tensor, Image.Image, Image.Image]]:
    ## Get PIL images

    if not args.demo:
        image_output = image_process_with_path(args)
    else:
        image_output = image_process_with_image(args, content_image, style_image)

    if image_output is None:
        return None

    content_image_pil, style_images_pil = image_output

    ## Transform images to tensors

    content_transforms = get_transform_function(
        target_size=args.content_image_size, normalize=True
    )
    style_transforms = get_transform_function(
        target_size=args.style_image_size, normalize=True
    )

    # Apply the transform to the content image
    content_image = content_transforms(content_image_pil)[None, :]
    # Apply the transform to the style image
    style_image = style_transforms(style_images_pil)[None, :]

    return content_image, style_image, content_image_pil, style_images_pil


def load_fontdiffuser_pipeline(args):
    # Load the model state_dict
    unet = build_unet(args=args)
    unet.load_state_dict(torch.load(f"{args.ckpt_dir}/unet.pth"))
    style_encoder = build_style_encoder(args=args)
    style_encoder.load_state_dict(torch.load(f"{args.ckpt_dir}/style_encoder.pth"))
    content_encoder = build_content_encoder(args=args)
    content_encoder.load_state_dict(torch.load(f"{args.ckpt_dir}/content_encoder.pth"))
    model = FontDiffuserModelDPM(
        unet=unet, style_encoder=style_encoder, content_encoder=content_encoder
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


def sampling(args, pipe, content_image=None, style_image=None):
    if args.save_image:
        os.makedirs(args.save_image_dir, exist_ok=True)

        # saving sampling config
        save_args_to_yaml(
            args=args, output_file=f"{args.save_image_dir}/sampling_config.yaml"
        )

    if type(args.seed) is int:
        set_seed(seed=args.seed)

    image_process_output = image_process(
        args=args, content_image=content_image, style_image=style_image
    )

    if image_process_output == None:
        print(
            f"The content_character you provided is not in the ttf. \
                Please change the content_character or you can change the ttf."
        )
        return None

    content_image, style_image, content_image_pil, _ = image_process_output

    with torch.no_grad():
        content_image = content_image.to(args.device)
        style_image = style_image.to(args.device)
        print(f"Sampling by DPM-Solver++ ......")
        start = time.time()
        images = pipe.generate(
            content_images=content_image,
            style_images=style_image,
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
