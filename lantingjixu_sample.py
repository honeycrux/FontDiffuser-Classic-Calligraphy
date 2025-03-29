# This script is provided by the FYP24 project group.
# This is the driver code for configuring and invoking the sampling process, which can be used in place of scripts/sample_content_character.sh.
# The ttf path, save path, text-to-generate path, and style image path can be configured in the main function.
# For example, to generate the entire lantingjixu text, use the whole lantingjixu text (data_lantingjixu/lantingjixu.txt) as the text-to-generate file.

import random
from typing import Optional
from sample import (
    arg_parse, 
    sampling,
    load_fontdiffuser_pipeline,
)
import os
import time
import torch

def load_text_to_generate(file_path: str):
    with open(file_path, 'r', encoding='utf-8') as text_file:
        text = text_file.read()
        characters = list(set(text))
        return characters

def load_essential_args(
        args,
        ckpt_dir: str,
        guidance_scale: float = 7.5,
    ):
    # essential args are the arguments that are required to run load_fontdiffuser_pipeline
    # which includes arguments required to build the model and its components

    args.guidance_type = 'classifier-free'

    args.device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")

    args.ckpt_dir = ckpt_dir
    args.guidance_scale = guidance_scale

    return args

def run_fontdiffuser(
        args,
        pipe,
        content_image_path: Optional[str],
        character: Optional[str],
        style_image_path: str,
        save_image_dir: str,
        ttf_path: str,
        sampling_step: int = 20,
        batch_size: int = 1,
        seed: Optional[int] = None,
    ):
    args.method = 'multistep'
    args.algorithm_type = 'dpmsolver++'

    args.demo = False
    args.save_image = False

    args.content_image_path = content_image_path
    args.character_input = False if content_image_path is not None else True
    args.content_character = character
    args.style_image_path = style_image_path
    args.save_image_dir = save_image_dir
    args.ttf_path = ttf_path
    args.sampling_step = sampling_step
    args.batch_size = batch_size

    args.seed = seed if type(seed) is int else random.randint(0, 10000)

    out_image = sampling(
        args=args,
        pipe=pipe,
        content_image=None,
        style_images=None,
    )
    return out_image

def main():
    args = arg_parse()

    ckpt_dir = 'ckpt/'
    ttf_path = 'ttf/SourceHanSerifTC-VF.ttf'
    save_image_dir = 'outputs/style_rec'
    style_image_path = 'data_lantingjixu/train/TargetImage/lan'
    seed = 0

    # load characters to generate
    text_to_generate_path = 'lantingjixu_test.txt'
    characters = load_text_to_generate(text_to_generate_path)

    # load fontdiffuser pipeline
    load_essential_args(
        args=args,
        ckpt_dir=ckpt_dir,
    )
    pipe = load_fontdiffuser_pipeline(args=args)

    total_time = 0
    total_sample = 0

    no_existence_check = True

    for i, character in enumerate(characters):
        if not no_existence_check and os.path.exists(f'{args.save_image_dir}/{character}.png'):
            print(f'[{i+1}/{len(characters)}] {args.save_image_dir}/{character}.png already exists')
        else:
            start_time = time.time()
            out_image = run_fontdiffuser(
                args=args,
                pipe=pipe,
                content_image_path=None,
                character=character,
                style_image_path=style_image_path,
                save_image_dir=save_image_dir,
                ttf_path=ttf_path,
                seed=seed,
            )
            assert out_image is not None
            out_image.save(f'{args.save_image_dir}/{character}.png')
            end_time = time.time()

            print(f"Finish the sampling process, costing time {end_time - start_time}s")
            total_time += end_time - start_time
            total_sample += 1
            print(f'[{i+1}/{len(characters)}] created {args.save_image_dir}/{character}.png')

    print(f"Total sampling time: {total_time}s")
    print(f"Total sampling: {total_sample}")
    print(f"Average sampling time: {0 if total_sample == 0 else total_time/total_sample}s")

if __name__ == '__main__':
    main()
