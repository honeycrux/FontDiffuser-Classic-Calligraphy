# This script is provided by the FYP24 project group.
# This is the driver code for configuring and invoking the sampling process, which can be used in place of scripts/sample_content_character.sh.
# The ckpt dir, ttf path, save image dir, style image dir, seed, title data path, and text data path can be configured in the main function.
# For example, to generate the entire lantingjixu text, use the whole lantingjixu text (data_lantingjixu/lantingjixu_text.txt) as the text data path.

import os
import random
import time
from collections import defaultdict
from typing import Optional

import torch

from sample import arg_parse, load_fontdiffuser_pipeline, sampling


def load_text(file_path: str):
    with open(file_path, "r", encoding="utf-8") as text_file:
        text = text_file.read()
        return text


def get_file_names(characters: str):
    word_count = defaultdict(lambda: 0)
    file_names = []
    for character in characters:
        seq = word_count[character]
        if seq == 0:
            file_names.append(f"{character}")
        else:
            file_names.append(f"{character}+{seq}")
        word_count[character] += 1
    return file_names


def load_essential_args(
    args,
    ckpt_dir: str,
    guidance_scale: float = 7.5,
):
    # essential args are the arguments that are required to run load_fontdiffuser_pipeline
    # which includes arguments required to build the model and its components

    args.guidance_type = "classifier-free"

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
    num_inference_steps: int = 20,
    batch_size: int = 1,
    seed: Optional[int] = None,
):
    args.method = "multistep"
    args.algorithm_type = "dpmsolver++"

    args.demo = False
    args.save_image = False

    args.content_image_path = content_image_path
    args.character_input = False if content_image_path is not None else True
    args.content_character = character
    args.style_image_path = style_image_path
    args.save_image_dir = save_image_dir
    args.ttf_path = ttf_path
    args.num_inference_steps = num_inference_steps
    args.batch_size = batch_size

    args.seed = seed if type(seed) is int else random.randint(0, 10000)

    out_image = sampling(
        args=args,
        pipe=pipe,
        content_image=None,
        style_image=None,
    )
    return out_image


def main():
    args = arg_parse()

    ckpt_dir = "ckpt/"
    ttf_path = "ttf/SourceHanSerifTC-VF.ttf"
    save_image_dir = "outputs/"
    style_image_dir = "data_lantingjixu/all/TargetImage/lan"
    seed = None

    require_title = True  # Whether to include the title

    title_data_path = (
        "data_lantingjixu/lantingjixu_title.txt"  # Set the path to the title
    )
    text_data_path = "data_lantingjixu/lantingjixu_text.txt"  # Set the path to the text

    title_text = load_text(title_data_path) if require_title else ""
    text_text = load_text(text_data_path)

    combined_text = title_text + text_text
    file_names = get_file_names(combined_text)

    # load fontdiffuser pipeline
    load_essential_args(
        args=args,
        ckpt_dir=ckpt_dir,
    )
    pipe = load_fontdiffuser_pipeline(args=args)

    total_time = 0
    total_sample = 0

    no_existence_check = True

    style_images = [f"{style_image_dir}/{img}" for img in os.listdir(style_image_dir)]

    for i, (character, file_name) in enumerate(zip(combined_text, file_names)):
        if not no_existence_check and os.path.exists(
            f"{save_image_dir}/{character}.png"
        ):
            print(
                f"[{i+1}/{len(combined_text)}] {save_image_dir}/{character}.png already exists"
            )
        else:
            start_time = time.time()

            # One-shot: choose a random style image
            style_image_path = random.choice(style_images)

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
            out_image.save(f"{save_image_dir}/{file_name}.png")
            end_time = time.time()

            print(f"Image generated (sampled) in {end_time - start_time}s")
            total_time += end_time - start_time
            total_sample += 1
            print(
                f"[{i+1}/{len(combined_text)}] Created {save_image_dir}/{file_name}.png"
            )

    print(f"Total sampling time: {total_time}s")
    print(f"Total sampling: {total_sample}")
    print(
        f"Average sampling time: {0 if total_sample == 0 else total_time/total_sample}s"
    )


if __name__ == "__main__":
    main()
