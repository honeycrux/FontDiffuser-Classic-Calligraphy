import os
import cv2
import time
import random
import numpy as np
from PIL import Image
from pathlib import Path

import torch
import torchvision.transforms as transforms
from accelerate.utils import set_seed

from src import (FontDiffuserDPMPipeline,
                 FontDiffuserModelDPM,
                 build_ddpm_scheduler,
                 build_unet,
                 build_content_encoder,
                 build_style_encoder,
                 build_k_feature_extractor)
from utils import (ttf2im,
                   load_ttf,
                   is_char_in_font,
                   save_args_to_yaml,
                   save_single_image,
                   save_image_with_content_style)


def arg_parse():
    from configs.fontdiffuser import get_parser

    parser = get_parser()   # Get the parser
    parser.add_argument("--ckpt_dir", type=str, default=None)   # The checkpoint directory
    parser.add_argument("--demo", action="store_true")          # If in demo mode
    parser.add_argument("--controlnet", type=bool, default=False,   # If use controlnet
                        help="If in demo mode, the controlnet can be added.")   # If in demo mode, the instructpix2pix can be added.
    parser.add_argument("--character_input", action="store_true")   # If the input is character
    parser.add_argument("--content_character", type=str, default=None)      # The content character
    parser.add_argument("--content_image_path", type=str, default=None)     # The content image path
    parser.add_argument("--style_image_path", type=str, default=None)       # The style image path
    parser.add_argument("--save_image", action="store_true")        # If save the image
    parser.add_argument("--save_image_dir", type=str, default=None,         # The saving directory
                        help="The saving directory.")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")  # The device (CPU or GPU)
    parser.add_argument("--ttf_path", type=str, default="ttf/KaiXinSongA.ttf")      # The ttf path
    args = parser.parse_args()              # Parse the arguments
    style_image_size = args.style_image_size                # The style image size
    content_image_size = args.content_image_size            # The content image size
    args.style_image_size = (style_image_size, style_image_size)        # convert style image size to tuple, image width and height are the same
    args.content_image_size = (content_image_size, content_image_size)      # convert content image size to tuple, image width and height are the same

    return args


def image_process(args, content_image=None, style_images=None):
    if not args.demo:       # If not in demo mode
        # Read content image and style image
        if args.character_input:    # If the input is character
            assert args.content_character is not None, "The content_character should not be None."
            if not is_char_in_font(font_path=args.ttf_path, char=args.content_character):   # If the character is not in the font
                return None, None
            font = load_ttf(ttf_path=args.ttf_path)     # Load the ttf
            content_image = ttf2im(font=font, char=args.content_character)      # Convert the ttf to image
            content_image_pil = content_image.copy()        # Copy the content image
        else:                       # If the input is image
            content_image = Image.open(args.content_image_path).convert('RGB')      # Open the content image
            content_image_pil = None        # The content image of the PIL format
        style_images_dir = Path(args.style_image_path)      # The style image directory
        style_images = []       # The style images
        for style_image_path in style_images_dir.iterdir():     # Iterate the style image directory
            if style_image_path.is_file():      # If the style image path is a file
                style_images.append(Image.open(style_image_path).convert('RGB'))        # Open the style image and append it to the style images
        # style_images = Image.open(args.style_image_path).convert('RGB')
    else:                   # If in demo mode
        assert style_images is not None, "The style image should not be None."
        if args.character_input:
            assert args.content_character is not None, "The content_character should not be None."
            if not is_char_in_font(font_path=args.ttf_path, char=args.content_character):
                return None, None
            font = load_ttf(ttf_path=args.ttf_path)
            content_image = ttf2im(font=font, char=args.content_character)
        else:
            assert content_image is not None, "The content image should not be None."
        content_image_pil = None
        
    ## Dataset transform
    content_inference_transforms = transforms.Compose(
        # Resize the image to the target size
        [transforms.Resize(args.content_image_size, \
                            interpolation=transforms.InterpolationMode.BILINEAR),
                            # Convert the image to a PyTorch tensor
            transforms.ToTensor(),
            # Normalize the image
            transforms.Normalize([0.5], [0.5])])
    # Style image transform
    style_inference_transforms = transforms.Compose(
        [transforms.Resize(args.style_image_size, \
                           interpolation=transforms.InterpolationMode.BILINEAR),
         transforms.ToTensor(),
         transforms.Normalize([0.5], [0.5])])
    # Apply the transform to the content image
    content_image = content_inference_transforms(content_image)[None, :]
    # Apply the transform to the style image
    style_images = [style_inference_transforms(style_image)[None, :] for style_image in style_images]

    return content_image, style_images, content_image_pil

def load_fontdiffuer_pipeline(args):
    # Load the model state_dict
    unet = build_unet(args=args)        # Build the unet model
    unet.load_state_dict(torch.load(f"{args.ckpt_dir}/unet.pth"))       # Load the unet model state_dict
    style_encoder = build_style_encoder(args=args)          # Build the style encoder
    style_encoder.load_state_dict(torch.load(f"{args.ckpt_dir}/style_encoder.pth"))         # Load the style encoder state_dict
    content_encoder = build_content_encoder(args=args)      # Build the content encoder
    content_encoder.load_state_dict(torch.load(f"{args.ckpt_dir}/content_encoder.pth"))         # Load the content encoder state_dict
    k_feature_extractor = build_k_feature_extractor(args=args)        # Build the k feature extractor
    k_feature_extractor.load_state_dict(torch.load(f"{args.ckpt_dir}/k_feature_extractor.pth"))       # Load the k feature extractor state_dict
    model = FontDiffuserModelDPM(           # Build the FontDiffuserModelDPM, do the __init__ function of FontDiffuserModelDPM
        unet=unet,
        style_encoder=style_encoder,
        content_encoder=content_encoder,
        k_feature_extractor=k_feature_extractor,)
    model.to(args.device)                   # Move the model to the device
    print("Loaded the model state_dict successfully!")

    # Load the training ddpm_scheduler.
    train_scheduler = build_ddpm_scheduler(args=args)       # Build the ddpm_scheduler
    print("Loaded training DDPM scheduler sucessfully!")

    # Load the DPM_Solver to generate the sample, do __init__ function of FontDiffuserDPMPipeline
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
    if not args.demo:   # If not in demo mode
        os.makedirs(args.save_image_dir, exist_ok=True)
        # saving sampling config
        save_args_to_yaml(args=args, output_file=f"{args.save_image_dir}/sampling_config.yaml")
    # Set the seed
    if args.seed:
        set_seed(seed=args.seed)
   #content_image_pil is the content image of the PIL format
    content_image, style_images, content_image_pil = image_process(args=args, 
                                                                  content_image=content_image, 
                                                                  style_images=style_images)
    if content_image == None:
        print(f"The content_character you provided is not in the ttf. \
                Please change the content_character or you can change the ttf.")
        return None

    with torch.no_grad():       # Disable the gradient calculation
        content_image = content_image.to(args.device)       # Move the content image to the device
        style_images = [style_image.to(args.device) for style_image in style_images]        # Move the style images to the device
        print(f"Sampling by DPM-Solver++ ......")
        start = time.time()
        # Generate the image
        images = pipe.generate(     #call the generate function of FontDiffuserDPMPipeline in pipeline_dpm_solver.py
            content_images=content_image,
            style_images=style_images,
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
            correcting_x0_fn=args.correcting_x0_fn)
        end = time.time()

        if args.save_image:
            print(f"Saving the image ......")
            save_single_image(save_dir=args.save_image_dir, image=images[0])
            if args.character_input:
                save_image_with_content_style(save_dir=args.save_image_dir,
                                            image=images[0],
                                            content_image_pil=content_image_pil,
                                            content_image_path=None,
                                            style_image_path=args.style_image_path,
                                            resolution=args.resolution)
            else:
                save_image_with_content_style(save_dir=args.save_image_dir,
                                            image=images[0],
                                            content_image_pil=None,
                                            content_image_path=args.content_image_path,
                                            style_image_path=args.style_image_path,
                                            resolution=args.resolution)
            print(f"Finish the sampling process, costing time {end - start}s")
        return images[0]

# ControlNet
def load_controlnet_pipeline(args,
                             config_path="lllyasviel/sd-controlnet-canny", 
                             ckpt_path="runwayml/stable-diffusion-v1-5"):
    from diffusers import ControlNetModel, AutoencoderKL
    # load controlnet model and pipeline
    from diffusers import StableDiffusionControlNetPipeline, UniPCMultistepScheduler
    controlnet = ControlNetModel.from_pretrained(config_path, 
                                                 torch_dtype=torch.float16,
                                                 cache_dir=f"{args.ckpt_dir}/controlnet")
    print(f"Loaded ControlNet Model Successfully!")
    pipe = StableDiffusionControlNetPipeline.from_pretrained(ckpt_path, 
                                                             controlnet=controlnet, 
                                                             torch_dtype=torch.float16,
                                                             cache_dir=f"{args.ckpt_dir}/controlnet_pipeline")
    # faster
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()
    print(f"Loaded ControlNet Pipeline Successfully!")

    return pipe


def controlnet(text_prompt, 
               pil_image,
               pipe):
    image = np.array(pil_image)
    # get canny image
    image = cv2.Canny(image=image, threshold1=100, threshold2=200)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    canny_image = Image.fromarray(image)
    
    seed = random.randint(0, 10000)
    generator = torch.manual_seed(seed)
    image = pipe(text_prompt, 
                 num_inference_steps=50, 
                 generator=generator, 
                 image=canny_image,
                 output_type='pil').images[0]
    return image


def load_instructpix2pix_pipeline(args,
                                  ckpt_path="timbrooks/instruct-pix2pix"):
    from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler
    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(ckpt_path, 
                                                                  torch_dtype=torch.float16)
    pipe.to(args.device)
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

    return pipe

def instructpix2pix(pil_image, text_prompt, pipe):
    image = pil_image.resize((512, 512))
    seed = random.randint(0, 10000)
    generator = torch.manual_seed(seed)
    image = pipe(prompt=text_prompt, image=image, generator=generator, 
                 num_inference_steps=20, image_guidance_scale=1.1).images[0]

    return image


if __name__=="__main__":
    args = arg_parse()
    
    # load fontdiffuser pipeline
    pipe = load_fontdiffuer_pipeline(args=args)
    out_image = sampling(args=args, pipe=pipe)
