import random
import os
import time
import yaml
from pathlib import Path

from PIL import Image
import torch
import torchvision.transforms as TF

from sample import (arg_parse, 
                    sampling,
                    load_fontdiffuer_pipeline)
from lantingjixu_performance import LantingjixuPerformance

def run_fontdiffuer_demo(args,
                    pipe,
                    content_image, 
                    character, 
                    style_images,
                    sampling_step,
                    guidance_scale,
                    batch_size,
                    seed,
                    few_shot):
    args.demo = True
    args.character_input = False if content_image is not None else True
    args.content_character = character
    args.sampling_step = sampling_step
    args.guidance_scale = guidance_scale
    args.batch_size = batch_size
    args.seed = seed if type(seed) is int else random.randint(0, 10000)
    out_image = sampling(
        args=args,
        pipe=pipe,
        content_image=content_image,
        style_images=style_images if few_shot else style_images[0])
    return out_image

def save_rounds_info(round_info: dict, output_dir: str):
    with open(f'{output_dir}/eval_info.yaml', 'w', encoding="utf-8") as yaml_file:
        yaml.dump(round_info, yaml_file, default_flow_style=False, allow_unicode=True)

def generate_rounds(n_rounds: int, round_size: int, num_style_image: int, dataset_files: list[Path]):
    available_character_choices = list(range(len(dataset_files)))

    rounds_info_saved = {}
    rounds_info_computation = {}

    for r in range(n_rounds):
        chosen_characters: list[int] = []
        chosen_styles: list[int] = []
        available_style_choices = list(range(len(dataset_files)))
        for _ in range(round_size):
            if len(available_character_choices) == 0:
                # replenish
                available_character_choices = list(range(len(dataset_files)))

            character = random.choice(available_character_choices)
            chosen_characters.append(character)
            available_character_choices.remove(character)
            available_style_choices.remove(character)

        for _ in range(num_style_image):
            style = random.choice(available_style_choices)
            chosen_styles.append(style)
            available_style_choices.remove(style)

        characters = [dataset_files[i] for i in chosen_characters]
        styles = [dataset_files[i] for i in chosen_styles]
        seed = random.randint(0, 10000)

        rounds_info_saved[r] = {
            'character': [file.stem for file in characters],
            'style': [file.stem for file in styles],
            'seed': seed,
        }

        rounds_info_computation[r] = {
            'character': characters,
            'style': styles,
            'seed': seed,
        }

    return rounds_info_saved, rounds_info_computation

def main():
    args = arg_parse()
    args.ckpt_dir = 'ckpt/'
    args.ttf_path = 'ttf/SourceHanSerifTC-VF.ttf'

    args.method = 'multistep'
    args.guidance_type = 'classifier-free'
    args.algorithm_type = 'dpmsolver++'

    args.device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")

    # evaluation parameters
    rounds = 13
    round_size = 13
    dataset_dir = 'lantingxu_resized/by_char'
    output_dir = 'outputs/eval_few_shot'
    few_shot = True

    pipe = load_fontdiffuer_pipeline(args=args)
    toTensor = TF.ToTensor()

    dataset_dir_path = Path(dataset_dir)
    dataset_files = [f for f in dataset_dir_path.iterdir()]

    total_time = 0
    total_rounds = 0

    overall_performance = LantingjixuPerformance(device=args.device)

    os.makedirs(output_dir, exist_ok=True)

    rounds_info_saved, rounds_info_computation = generate_rounds(
        n_rounds=rounds,
        round_size=round_size,
        num_style_image=1,
        dataset_files=dataset_files
    )
    save_rounds_info(rounds_info_saved, output_dir)

    for r in range(rounds):
        start_time = time.time()
        print(f"Round {r + 1} (Saves: {output_dir}/{r})")

        current_round_info = rounds_info_computation[r]
        character_files = current_round_info['character']
        style_files = current_round_info['style']
        seed = current_round_info['seed']

        os.makedirs(f'{output_dir}/{r}', exist_ok=True)

        character_images = [Image.open(f).convert('RGB') for f in character_files]
        style_images = [Image.open(f).convert('RGB') for f in style_files]
        output_images: list[Image.Image] = []
        round_performance = LantingjixuPerformance(device=args.device)

        for character_file in character_files:
            character = character_file.stem

            out_image = run_fontdiffuer_demo(args=args,
                                        pipe=pipe,
                                        content_image=None,
                                        character=character,
                                        style_images=style_images,
                                        sampling_step=20,
                                        guidance_scale=7.5,
                                        batch_size=1,
                                        seed=seed,
                                        few_shot=few_shot)

            out_image.save(f'{output_dir}/{r}/{character}.png')
            output_images.append(out_image)

        if output_images[0].size != character_images[0].size:
            output_images = [output_image.resize(character_images[0].size, Image.Resampling.BILINEAR) for output_image in output_images]

        output_image_batch = torch.stack([toTensor(i) for i in output_images])
        character_image_batch = torch.stack([toTensor(i) for i in character_images])

        round_performance.update(output_image_batch, character_image_batch)
        overall_performance.update(output_image_batch, character_image_batch)

        round_performance_result = round_performance.compute()
        rounds_info_saved[r]['round_performance'] = round_performance_result
        save_rounds_info(rounds_info_saved, output_dir)

        end_time = time.time()
        print(f"Finish the sampling-evaluation process, costing time {end_time - start_time}s")
        total_time += end_time - start_time
        total_rounds += 1

    overall_performance_result = overall_performance.compute()
    rounds_info_saved['overall_performance'] = overall_performance_result
    save_rounds_info(rounds_info_saved, output_dir)

    print("Evaluation finished. Overall performance:"
          f"fid: {overall_performance_result['fid']}, "
          f"ssim: {overall_performance_result['ssim']}, "
          f"lpips: {overall_performance_result['lpips']}, "
          f"l1: {overall_performance_result['l1']}")

    print(f"Total time: {total_time}s")
    print(f"Total # rounds: {total_rounds}")
    print(f"Average round time: {0 if total_rounds == 0 else total_time/total_rounds}s")

if __name__ == '__main__':
    main()