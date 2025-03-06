# This script is provided by the FYP24 project group.
# This script is for configuring and invoking the sampling process, which can be used in place of scripts/sample_content_character.sh.
# The ttf path, save path, text-to-generate path, and style image path can be configured in the main function.
# For example, to generate the entire lantingjixu text, use the whole lantingjixu text (lantingjixu_data/lantingjixu.txt) as the text-to-generate file.

import random
from sample import (arg_parse, 
                    sampling,
                    load_fontdiffuser_pipeline)
import os
import time
import torch

def load_text_to_generate(file_path):
    with open(file_path, 'r', encoding='utf-8') as text_file:
        text = text_file.read()
        characters = list(set(text))
        return characters

def run_fontdiffuser(args,
                     pipe,
                     content_image_path, 
                     character, 
                     style_image_path,
                     save_image_dir,
                     sampling_step,
                     guidance_scale,
                     batch_size,
                     seed):
    args.demo = False
    args.content_image_path = content_image_path
    args.style_image_path = style_image_path
    args.save_image_dir = save_image_dir
    args.character_input = False if content_image_path is not None else True
    args.content_character = character
    args.sampling_step = sampling_step
    args.guidance_scale = guidance_scale
    args.batch_size = batch_size
    args.seed = seed if type(seed) is int else random.randint(0, 10000)
    out_image = sampling(
        args=args,
        pipe=pipe,
        content_image=None,
        style_images=None)
    return out_image

def main():
    args = arg_parse()
    args.ckpt_dir = 'ckpt/'
    args.ttf_path = 'ttf/SourceHanSerifTC-VF.ttf'

    args.method = 'multistep'
    args.guidance_type = 'classifier-free'
    args.algorithm_type = 'dpmsolver++'

    args.save_image = False
    args.save_image_dir = 'outputs/style_reconst'

    args.device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")

    # load characters to generate
    text_to_generate_path = 'lantingjixu_test.txt'
    characters = load_text_to_generate(text_to_generate_path)

    # load fontdiffuser pipeline
    pipe = load_fontdiffuser_pipeline(args=args)

    total_time = 0
    total_sample = 0

    no_existence_check = True      # set to True to skip the existence check

    for i, character in enumerate(characters):
        if not no_existence_check and os.path.exists(f'{args.save_image_dir}/{character}.png'):
            print(f'[{i+1}/{len(characters)}] {args.save_image_dir}/{character}.png already exists')
        else:
            start_time = time.time()
            out_image = run_fontdiffuser(args=args,
                                         pipe=pipe,
                                         content_image_path=None,
                                         character=character,
                                         style_image_path='outputs/style_images',
                                         save_image_dir=args.save_image_dir,
                                         sampling_step=20,
                                         guidance_scale=7.5,
                                         batch_size=1,
                                         seed=0)
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
