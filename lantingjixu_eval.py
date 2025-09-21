# This script is provided by the FYP24 project group.
# This is the driver code to run whole evaluation process on the LantingjiXu dataset.
# It generates a test profile and runs sampling, then calculates the FID, SSIM, LPIPS, and L1 metrics.

import os
import random
import time
from pathlib import Path
from typing import Any, Optional

import torch
import torchvision.transforms as TF
import yaml
from PIL import Image

from sample import arg_parse, load_fontdiffuser_pipeline, sampling
from src.metrics.font_metrics import FontMetrics


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


def run_fontdiffuser_demo_mode(
    args,
    pipe,
    content_image: Optional[Image.Image],
    character: Optional[str],
    style_images: list[Image.Image],
    ttf_path: str,
    use_few_shot: bool,
    num_inference_steps: int = 20,
    batch_size: int = 1,
    seed: Optional[int] = None,
):
    args.method = "multistep"
    args.algorithm_type = "dpmsolver++"

    args.demo = True

    args.character_input = False if content_image is not None else True
    args.content_character = character
    args.num_inference_steps = num_inference_steps
    args.ttf_path = ttf_path
    args.batch_size = batch_size

    args.seed = seed if type(seed) is int else random.randint(0, 10000)

    sampling_args = dict[str, Any](
        args=args,
        pipe=pipe,
        content_image=content_image,
    )
    if use_few_shot:
        sampling_args["style_images"] = style_images
    else:
        sampling_args["style_image"] = style_images[0]

    out_image = sampling(**sampling_args)
    return out_image


def generate_single_test(num_style_image: int, dataset_files: list[Path]):
    # A test is run on every character in the dataset, each with a random selection of style images.

    test_info = {}

    for file_idx in range(len(dataset_files)):
        available_style_choices = list(range(len(dataset_files)))
        chosen_styles: list[int] = []

        available_style_choices.remove(file_idx)

        while len(chosen_styles) < num_style_image:
            style = random.choice(available_style_choices)
            chosen_styles.append(style)
            available_style_choices.remove(style)

        styles = [dataset_files[i] for i in chosen_styles]

        test_info[file_idx] = {
            "character": dataset_files[file_idx].name,
            "style": [file.name for file in styles],
        }

    return test_info


def create_test_profile(
    profile_dir: str,
    num_test_round: int,
    num_style_image: int,
    dataset_files: list[Path],
):
    # A profile is a collection of tests.

    os.makedirs(profile_dir, exist_ok=True)

    for test_idx in range(num_test_round):
        seed = random.randint(0, 10000)
        test_info = generate_single_test(
            num_style_image=num_style_image, dataset_files=dataset_files
        )
        test_configuration = {
            "index": test_idx,
            "test_info": test_info,
            "seed": seed,
        }
        with open(
            f"{profile_dir}/test_{test_idx}.yaml", "w", encoding="utf-8"
        ) as yaml_file:
            yaml.dump(
                test_configuration,
                yaml_file,
                default_flow_style=False,
                allow_unicode=True,
            )

    print(f"[Eval] Test profile created at {profile_dir} ({num_test_round} tests)")


def load_test_profile(profile_dir: str):
    # Load a profile from a directory.
    # The profile is a collection of tests.

    if not os.path.exists(profile_dir):
        return None

    profile_files = [f for f in Path(profile_dir).iterdir()]

    if len(profile_files) == 0:
        return None

    profile_info_map = {}
    for profile_file in profile_files:
        with open(profile_file, "r", encoding="utf-8") as yaml_file:
            test_configuration = yaml.load(yaml_file, Loader=yaml.FullLoader)
            profile_info_map[test_configuration["index"]] = test_configuration

    profile_info = []
    test_idx = 0
    while test_idx in profile_info_map:
        profile_info.append(profile_info_map[test_idx])
        test_idx += 1

    if len(profile_info_map) != len(profile_info):
        raise ValueError(
            "Erorr when loading a test profile: test indices are not continuous starting from 0"
        )

    return profile_info


def save_results(result_info: dict, output_dir: str):
    with open(f"{output_dir}/eval_results.yaml", "w", encoding="utf-8") as yaml_file:
        yaml.dump(result_info, yaml_file, default_flow_style=False, allow_unicode=True)


def parse_target_image_name(target_image_name: str):
    # Input Format: style+content[+optional-suffix]
    target_components = target_image_name.split("+")
    style = target_components[0]
    content = target_components[1]
    return style, content


def main():
    args = arg_parse()
    ckpt_dir = "ckpt/"
    ttf_path = "ttf/SourceHanSerifTC-VF.ttf"

    ### Evaluation configuration ###

    # Few-shot or single-shot (depends on the current model used)
    # Note: If use few-shot, the number of style images follows the num_style_image configuration in the test profile.
    # If use single-shot, only the first style image will be used.
    use_few_shot = False

    # Dataset location
    dataset_dir = "data_lantingjixu/test/TargetImage/lan"

    # Configure the test profile. If the profile does not exist, it will be created.
    # Note: If you use an existing profile, please make sure the dataset is the same as the one used to create the profile.
    test_profile_dir = "outputs/test-profile-2025-02-01"
    num_test_round = 10
    num_style_image = 5

    # If the profile already exists, set this to True.
    # This prevents the evaluation process from regenerating the profile if you want reproducible results.
    expect_existing_profile = True

    # Results location
    results_output_dir = "outputs/eval"

    ### Part 1: Load/Generate the test profile ###

    dataset_dir_path = Path(dataset_dir)
    dataset_files = [f for f in dataset_dir_path.iterdir()]

    profile_info = load_test_profile(profile_dir=test_profile_dir)
    print()
    if profile_info is None:
        if expect_existing_profile:
            raise ValueError(
                "Test profile expected but does not exist. \n"
                "Did you mean to include an existing profile in the test_profile_dir? \n"
                "If you want to generate a new profile, set expect_existing_profile to False."
            )

        print(f"[Eval] No test profile found. Creating a new test profile")
        create_test_profile(
            profile_dir=test_profile_dir,
            num_test_round=num_test_round,
            num_style_image=num_style_image,
            dataset_files=dataset_files,
        )
        profile_info = load_test_profile(profile_dir=test_profile_dir)
        assert profile_info is not None
        print(
            f"[Eval] Test profile loaded from {test_profile_dir} ({len(profile_info)} tests)"
        )
    else:
        print(
            f"[Eval] Test profile loaded from {test_profile_dir} ({len(profile_info)} tests)"
        )
    print()

    ### Part 2: Run the evaluation process ###

    print("[Eval] Evaluation begins")
    print()

    load_essential_args(
        args=args,
        ckpt_dir=ckpt_dir,
    )
    pipe = load_fontdiffuser_pipeline(args=args)
    toTensor = TF.ToTensor()

    total_time = 0
    total_rounds = 0

    overall_performance = FontMetrics(device=args.device)

    test_results = {}

    os.makedirs(results_output_dir, exist_ok=True)
    save_results(test_results, results_output_dir)

    total_tests = len(profile_info)

    for test_idx, test_info in enumerate(profile_info):
        start_time = time.time()
        print()
        print(f"[Eval] {total_tests - test_idx} tests remaining")
        print(
            f"[Eval] Test {test_idx} begins (Saving to {results_output_dir}/{test_idx})"
        )
        print()

        seed = test_info["seed"]
        test_performance = FontMetrics(device=args.device)

        os.makedirs(f"{results_output_dir}/{test_idx}", exist_ok=True)

        total_files = len(test_info["test_info"])

        for file_idx, file_info in enumerate(test_info["test_info"].values()):
            print(
                f"[{test_idx + 1}/{total_tests}][{file_idx + 1}/{total_files}] ", end=""
            )

            character_file = Path(f"{dataset_dir}/{file_info['character']}")
            style_files = [
                Path(f"{dataset_dir}/{style}") for style in file_info["style"]
            ]

            character_image = Image.open(character_file).convert("RGB")
            style_images = [Image.open(f).convert("RGB") for f in style_files]

            _, character = parse_target_image_name(character_file.stem)

            out_image = run_fontdiffuser_demo_mode(
                args=args,
                pipe=pipe,
                content_image=None,
                character=character,
                style_images=style_images,
                ttf_path=ttf_path,
                use_few_shot=use_few_shot,
                seed=seed,
            )

            assert out_image is not None

            out_image.save(f"{results_output_dir}/{test_idx}/{character}.png")

            if out_image.size != character_image.size:
                out_image = out_image.resize(
                    character_image.size, Image.Resampling.BILINEAR
                )

            output_image_batch = torch.stack([toTensor(out_image)])
            character_image_batch = torch.stack([toTensor(character_image)])

            test_performance.update(output_image_batch, character_image_batch)
            overall_performance.update(output_image_batch, character_image_batch)

        test_performance_result = test_performance.compute()
        test_results[test_idx] = test_performance_result
        save_results(test_results, results_output_dir)

        end_time = time.time()
        round_time = end_time - start_time
        total_time += round_time
        total_rounds += 1

        print()
        print(
            f"[Eval] Sampling and evaluation for test {test_idx} ended, costing time {round_time}s"
        )

    overall_performance_result = overall_performance.compute()
    average_round_time = 0 if total_rounds == 0 else total_time / total_rounds

    fid_values = torch.tensor(
        [test_result["fid"] for test_result in test_results.values()]
    )
    ssim_values = torch.tensor(
        [test_result["ssim"] for test_result in test_results.values()]
    )
    lpips_values = torch.tensor(
        [test_result["lpips"] for test_result in test_results.values()]
    )
    l1_values = torch.tensor(
        [test_result["l1"] for test_result in test_results.values()]
    )

    mean_result = {
        "fid": fid_values.mean().item(),
        "ssim": ssim_values.mean().item(),
        "lpips": lpips_values.mean().item(),
        "l1": l1_values.mean().item(),
    }

    std_result = {
        "fid": fid_values.std().item(),
        "ssim": ssim_values.std().item(),
        "lpips": lpips_values.std().item(),
        "l1": l1_values.std().item(),
    }

    test_results["mean"] = mean_result
    test_results["std"] = std_result
    test_results["overall_performance"] = overall_performance_result
    test_results["total_time"] = total_time
    test_results["total_rounds"] = total_rounds
    test_results["average_round_time"] = average_round_time

    save_results(test_results, results_output_dir)

    # Print overall test results

    print()
    print("[Eval] Evaluation finished")
    print(
        "Overall performance: \n"
        f"\tfid: {overall_performance_result['fid']} (mean: {mean_result['fid']}, std: {std_result['fid']}), \n"
        f"\tssim: {overall_performance_result['ssim']} (mean: {mean_result['ssim']}, std: {std_result['ssim']}), \n"
        f"\tlpips: {overall_performance_result['lpips']} (mean: {mean_result['lpips']}, std: {std_result['lpips']}), \n"
        f"\tl1: {overall_performance_result['l1']} (mean: {mean_result['l1']}, std: {std_result['l1']})"
    )

    print(f"Total time: {total_time}s")
    print(f"Total # tests: {total_rounds}")
    print(f"Average test time: {average_round_time}s")


if __name__ == "__main__":
    main()
