# This script is provided by the FYP24 project group.
# This is the driver code to run whole evaluation process on the ZHUOJG dataset which has three categories.
# It generates a test profile and runs sampling, then calculates the FID, SSIM, LPIPS, and L1 metrics.

from collections import defaultdict
import os
import random
import time
from pathlib import Path
from typing import Any, Optional
import math
import statistics

import torch
import torchvision.transforms as TF
import yaml
from PIL import Image

from sample import ReferenceImage, ReferenceImageList, SourceImage, arg_parse, load_fontdiffuser_pipeline, sampling
from src.metrics.font_metrics import FontMetrics


class FileInfo:
    font_name: str
    character: str
    path: Path

    def __init__(self, path: Path):
        style, character = FileInfo._parse_target_image_name(path.stem)
        assert len(character) == 1, f"Character length for {path} should be 1"
        self.font_name = style
        self.character = character
        self.path = path

    @staticmethod
    def _parse_target_image_name(target_image_name: str):
        # Input Format: style+content[+optional-suffix]
        target_components = target_image_name.split("+")
        style = target_components[0]
        content = target_components[1]
        return style, content


class PickerFontInfo:
    name: str
    category: str
    style: str
    files: list[FileInfo]
    count: int

    def __init__(self, name: str, files: list[Path]):
        category, style = PickerFontInfo._parse_font_name(name)

        self.name = name
        self.category = category
        self.style = style
        self.files = [FileInfo(path=file_path) for file_path in files]
        for file_info in self.files:
            assert file_info.font_name == self.name, f"File {file_info.path} has font name {file_info.font_name} which does not match with the font info name {self.name}"
        self.count = len(files)

    @staticmethod
    def _parse_font_name(font_name: str):
        # Input format: category-style
        font_components = font_name.split("-")
        category = font_components[0]
        style = font_components[1]
        return category, style

    def choose_files(self, num_files_to_choose: int, files_to_skip: list[FileInfo]) -> list[FileInfo]:
        candidate_files = [file_info for file_info in self.files if file_info not in files_to_skip]
        if len(candidate_files) <= num_files_to_choose:
            return candidate_files
        else:
            return random.sample(candidate_files, num_files_to_choose)


class TestInstance:
    data_directory: Path
    character: str
    reference_images: list[FileInfo]
    actual_image: FileInfo

    def __init__(self, data_directory: Path, character: str, reference_images: list[FileInfo], actual_image: FileInfo):
        self.data_directory = data_directory
        self.character = character
        assert len(character) == 1, f"Character length for {character} should be 1"
        self.reference_images = reference_images
        self.actual_image = actual_image

    @staticmethod
    def read(test_instance_info: dict, data_directory: Path) -> "TestInstance":
        character = test_instance_info["character"]
        assert type(character) is str, f"Character should be a string, but got {character} with type {type(character)}"

        reference_images: list[FileInfo] = []
        for reference in test_instance_info["reference_images"]:
            assert type(reference) is str, f"Reference image should be a string, but got {reference} with type {type(reference)}"
            reference_images.append(FileInfo(data_directory / reference))

        actual_image = test_instance_info["actual_image"]
        assert type(actual_image) is str, f"Actual image should be a string, but got {actual_image} with type {type(actual_image)}"
        actual_image = FileInfo(data_directory / actual_image)

        return TestInstance(data_directory=data_directory, character=character, reference_images=reference_images, actual_image=actual_image)

    def to_dict(self) -> dict[str, Any]:
        return {
            "character": self.character,
            "reference_images": [file_info.path.relative_to(self.data_directory).as_posix() for file_info in self.reference_images],
            "actual_image": self.actual_image.path.relative_to(self.data_directory).as_posix(),
        }


class PickerCategory:
    category: str
    fonts: list[PickerFontInfo]
    category_limit: int
    per_font_limit: int

    def __init__(self, category: str, fonts: list[PickerFontInfo], category_limit: int):
        self.category = category
        self.fonts = fonts
        self.category_limit = category_limit
        self.per_font_limit = math.floor(category_limit / len(fonts))

    def pick_test_instances(self, num_reference_images: int, data_directory: Path) -> list[TestInstance]:
        test_instances: list[TestInstance] = []
        for font in self.fonts:
            chosen_files = font.choose_files(self.per_font_limit, files_to_skip=[])
            test_instances.extend([
                TestInstance(
                    data_directory=data_directory,
                    character=file_info.character,
                    reference_images=font.choose_files(num_reference_images, files_to_skip=[file_info]),
                    actual_image=file_info,
                )
                for file_info in chosen_files
            ])

        return test_instances


class TestSetFilePicker:
    categories: list[PickerCategory]

    def __init__(self, categories: list[PickerCategory]):
        self.categories = categories

    @staticmethod
    def load_from_dir(dataset_dir: Path, category_limit: int) -> "TestSetFilePicker":
        by_category: defaultdict[str, list[PickerFontInfo]] = defaultdict(lambda: [])

        for font_dir in dataset_dir.iterdir():
            if not font_dir.is_dir():
                continue
            font_name = font_dir.name
            font_files = list(font_dir.iterdir())
            font_info = PickerFontInfo(name=font_name, files=font_files)
            category_name = font_info.category
            by_category[category_name].append(font_info)

        categories = []
        for category_name, fonts in by_category.items():
            category = PickerCategory(category=category_name, fonts=fonts, category_limit=category_limit)
            categories.append(category)

        return TestSetFilePicker(categories=categories)

class TestInfo:
    test_file: Path
    data_category: str
    font_category: str
    data_directory: Path
    seed: int
    test_instances: list[TestInstance]

    def __init__(self, test_file: Path, data_category: str, font_category: str, data_directory: Path, seed: int, test_instances: list[TestInstance]):
        self.test_file = test_file
        self.data_category = data_category
        self.font_category = font_category
        self.data_directory = data_directory
        self.seed = seed
        self.test_instances = test_instances

    def name(self) -> str:
        return f"{self.data_category}-{self.font_category}"

    @staticmethod
    def load_test(test_file: Path) -> "TestInfo":
        with open(test_file, "r", encoding="utf-8") as yaml_file:
            test_configuration = yaml.load(yaml_file, Loader=yaml.FullLoader)

        data_category = test_configuration["data_category"]
        assert type(data_category) is str, f"Test data_category should be str, but got {type(data_category)}"

        font_category = test_configuration["font_category"]
        assert type(font_category) is str, f"Test font_category should be str, but got {type(font_category)}"

        data_directory = test_configuration["data_directory"]
        assert type(data_directory) is str, f"Test data_directory should be str, but got {type(data_directory)}"
        data_directory = Path(data_directory)

        seed = test_configuration["seed"]
        assert type(seed) is int, f"Test seed should be int, but got {type(seed)}"

        test_instances = []
        for test_instance in test_configuration["test_instances"]:
            test_instances.append(TestInstance.read(test_instance, data_directory))

        return TestInfo(test_file=test_file, data_category=data_category, font_category=font_category, data_directory=data_directory, seed=seed, test_instances=test_instances)

    @staticmethod
    def generate_tests(test_profile_directory: Path, data_category: str, data_directory: Path, num_reference_images: int, category_limit: int) -> list["TestInfo"]:
        tests: list["TestInfo"] = []

        picker = TestSetFilePicker.load_from_dir(dataset_dir=data_directory, category_limit=category_limit)

        for category in picker.categories:
            category_test_instances = category.pick_test_instances(num_reference_images=num_reference_images, data_directory=data_directory)
            test_file = test_profile_directory / f"{data_category}-{category.category}.yaml"
            test_info = TestInfo(
                test_file=test_file,
                data_category=data_category,
                font_category=category.category,
                data_directory=data_directory,
                seed=random.randint(0, 10000),
                test_instances=category_test_instances,
            )
            tests.append(test_info)

        for test_info in tests:
            test_info._write_to_file()

        return tests

    def _write_to_file(self):
        test_configuration = {
            "data_category": self.data_category,
            "font_category": self.font_category,
            "data_directory": self.data_directory.as_posix(),
            "seed": self.seed,
            "test_instances": [test_instance.to_dict() for test_instance in self.test_instances],
        }

        with open(self.test_file, "w", encoding="utf-8") as yaml_file:
            yaml.dump(
                test_configuration,
                yaml_file,
                default_flow_style=False,
                allow_unicode=True,
            )


class TestProfile:
    tests: list[TestInfo]

    def __init__(self, tests: list[TestInfo]):
        self.tests = tests

    @staticmethod
    def load_test_profile(test_profile_directory: str) -> Optional["TestProfile"]:
        # Load a profile from a directory.
        # The profile is a collection of tests.

        if not os.path.exists(test_profile_directory):
            return None

        profile_files = [f for f in Path(test_profile_directory).iterdir()]

        if len(profile_files) == 0:
            return None

        test_list: list[TestInfo] = []
        for profile_file in profile_files:
            if profile_file.suffix != ".yaml":
                continue

            test_info = TestInfo.load_test(profile_file)
            if test_info is not None:
                test_list.append(test_info)

        return TestProfile(tests=test_list)

    @staticmethod
    def create_test_profile(
            test_profile_directory: Path,
            sfsc_data_directory: Path,
            sfuc_data_directory: Path,
            ufsc_data_directory: Path,
            ufuc_data_directory: Path,
            category_limit: int,
            num_reference_images: int
    ) -> "TestProfile":
        os.makedirs(test_profile_directory, exist_ok=True)

        tests_for_sfsc = TestInfo.generate_tests(
            test_profile_directory=test_profile_directory,
            data_category="sfsc",
            data_directory=sfsc_data_directory,
            num_reference_images=num_reference_images,
            category_limit=category_limit,
        )

        tests_for_sfuc = TestInfo.generate_tests(
            test_profile_directory=test_profile_directory,
            data_category="sfuc",
            data_directory=sfuc_data_directory,
            num_reference_images=num_reference_images,
            category_limit=category_limit,
        )

        tests_for_ufsc = TestInfo.generate_tests(
            test_profile_directory=test_profile_directory,
            data_category="ufsc",
            data_directory=ufsc_data_directory,
            num_reference_images=num_reference_images,
            category_limit=category_limit,
        )

        tests_for_ufuc = TestInfo.generate_tests(
            test_profile_directory=test_profile_directory,
            data_category="ufuc",
            data_directory=ufuc_data_directory,
            num_reference_images=num_reference_images,
            category_limit=category_limit,
        )

        tests = tests_for_sfsc + tests_for_sfuc + tests_for_ufsc + tests_for_ufuc

        print(f"[Eval] Test profile created at {test_profile_directory} ({len(tests)} tests)")

        return TestProfile(tests=tests)


class TestResult:
    data_category: str
    font_category: str
    fid: float
    ssim: float
    lpips: float
    l1: float
    total_runtime: float
    mean_runtime: float

    def __init__(self, data_category: str, font_category: str, fid: float, ssim: float, lpips: float, l1: float, runtime_secs: list[float]):
        self.data_category = data_category
        self.font_category = font_category
        self.fid = fid
        self.ssim = ssim
        self.lpips = lpips
        self.l1 = l1
        self.total_runtime = sum(runtime_secs)
        self.mean_runtime = statistics.mean(runtime_secs) if len(runtime_secs) > 0 else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "data_category": self.data_category,
            "font_category": self.font_category,
            "result": {
                "fid": self.fid,
                "ssim": self.ssim,
                "lpips": self.lpips,
                "l1": self.l1,
                "total_runtime": self.total_runtime,
                "average_runtime": self.mean_runtime,
            }
        }


class AnalyzeResult:
    name: str
    fid_mean: float
    fid_std: float
    ssim_mean: float
    ssim_std: float
    lpips_mean: float
    lpips_std: float
    l1_mean: float
    l1_std: float
    runtime_mean: float
    runtime_std: float

    def __init__(self, name: str, test_results: list[TestResult]):
        self.name = name

        fid_values = [res.fid for res in test_results]
        ssim_values = [res.ssim for res in test_results]
        lpips_values = [res.lpips for res in test_results]
        l1_values = [res.l1 for res in test_results]
        runtime_values = [res.mean_runtime for res in test_results]

        self.fid_mean = statistics.mean(fid_values) if len(fid_values) > 0 else 0.0
        self.fid_std = statistics.stdev(fid_values) if len(fid_values) > 1 else 0.0
        self.ssim_mean = statistics.mean(ssim_values) if len(ssim_values) > 0 else 0.0
        self.ssim_std = statistics.stdev(ssim_values) if len(ssim_values) > 1 else 0.0
        self.lpips_mean = statistics.mean(lpips_values) if len(lpips_values) > 0 else 0.0
        self.lpips_std = statistics.stdev(lpips_values) if len(lpips_values) > 1 else 0.0
        self.l1_mean = statistics.mean(l1_values) if len(l1_values) > 0 else 0.0
        self.l1_std = statistics.stdev(l1_values) if len(l1_values) > 1 else 0.0
        self.runtime_mean = statistics.mean(runtime_values) if len(runtime_values) > 0 else 0.0
        self.runtime_std = statistics.stdev(runtime_values) if len(runtime_values) > 1 else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "item": self.name,
            "result": {
                "fid_mean": self.fid_mean,
                "fid_std": self.fid_std,
                "ssim_mean": self.ssim_mean,
                "ssim_std": self.ssim_std,
                "lpips_mean": self.lpips_mean,
                "lpips_std": self.lpips_std,
                "l1_mean": self.l1_mean,
                "l1_std": self.l1_std,
                "runtime_mean": self.runtime_mean,
                "runtime_std": self.runtime_std,
            }
        }


class TestResults:
    directory: Path
    test_results: list[TestResult]
    analyze_results: list[AnalyzeResult]

    def __init__(self, directory: Path):
        os.makedirs(directory, exist_ok=True)
        self.directory = directory
        self.test_results = []
        self.analyze_results = []

    def add(self, test_result: TestResult):
        self.test_results.append(test_result)

    def analyze(self):
        by_font_category: dict[str, list[TestResult]] = defaultdict(lambda: [])
        by_data_category: dict[str, list[TestResult]] = defaultdict(lambda: [])

        for test_result in self.test_results:
            by_font_category[test_result.font_category].append(test_result)
            by_data_category[test_result.data_category].append(test_result)

        overall = AnalyzeResult(name="Overall", test_results=self.test_results)

        analyzed_by_font_category = [
            AnalyzeResult(name=f"Font Category: {font_category}", test_results=test_results)
            for font_category, test_results in by_font_category.items()
        ]

        analyzed_by_data_category = [
            AnalyzeResult(name=f"Data Category: {data_category}", test_results=test_results)
            for data_category, test_results in by_data_category.items()
        ]

        self.analyze_results = [overall] + analyzed_by_font_category + analyzed_by_data_category

    def save(self):
        eval_results = {
            "test_results": [test_result.to_dict() for test_result in self.test_results],
            "analyze_results": [analyze_result.to_dict() for analyze_result in self.analyze_results],
        }

        with open(self.directory / "eval_results.yaml", "w", encoding="utf-8") as yaml_file:
            yaml.dump(eval_results, yaml_file, default_flow_style=False, allow_unicode=True)


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
    content_image: Optional[SourceImage],
    character: Optional[str],
    style_images: Optional[ReferenceImageList],
    ttf_path: str,
    k_shot: int,
    num_inference_steps: int = 20,
    batch_size: int = 1,
    seed: Optional[int] = None,
):
    start_time = time.time()

    args.method = "multistep"
    args.algorithm_type = "dpmsolver++"

    args.demo = True

    args.character_input = False if content_image is not None else True
    args.content_character = character
    args.num_inference_steps = num_inference_steps
    args.ttf_path = ttf_path
    args.k_shot = k_shot
    args.batch_size = batch_size

    args.seed = seed if type(seed) is int else random.randint(0, 10000)

    out_image = sampling(
        args=args,
        pipe=pipe,
        content_image=content_image,
        style_images=style_images,
    )

    end_time = time.time()
    runtime_secs = end_time - start_time

    return out_image, runtime_secs


def main():
    args = arg_parse()
    ckpt_dir = "ckpt/"
    ttf_path = "ttf/SourceHanSerifTC-VF.ttf"

    ### Evaluation configuration ###

    # Configure the test profile. If the profile does not exist, it will be created.
    # Note: If you use an existing profile, please make sure the dataset is the same as the one used to create the profile.
    test_profile_directory = "outputs/test-profile-zhuojg-limited-144-25"
    category_limit = 144
    num_reference_images = 25
    sfsc_data_directory = "data/test/TestSet-SeenFont-SeenChar"
    sfuc_data_directory = "data/test/TestSet-SeenFont-UnseenChar"
    ufsc_data_directory = "data/test/TestSet-UnseenFont-SeenChar"
    ufuc_data_directory = "data/test/TestSet-UnseenFont-UnseenChar"

    # If the profile already exists, set this to True.
    # This prevents the evaluation process from regenerating the profile if you want reproducible results.
    expect_existing_profile = True

    # Results location
    test_results_dir = "outputs/eval-zhuojg"

    ### Part 1: Load/Generate the test profile ###

    profile_info = TestProfile.load_test_profile(test_profile_directory=test_profile_directory)
    print()
    if profile_info is None:
        if expect_existing_profile:
            raise ValueError(
                "Test profile expected but does not exist. \n"
                "Did you mean to include an existing profile in the test_profile_directory? \n"
                "If you want to generate a new profile, set expect_existing_profile to False."
            )

        print(f"[Eval] No test profile found. Creating a new test profile")

        profile_info = TestProfile.create_test_profile(
            test_profile_directory=Path(test_profile_directory),
            sfsc_data_directory=Path(sfsc_data_directory),
            sfuc_data_directory=Path(sfuc_data_directory),
            ufsc_data_directory=Path(ufsc_data_directory),
            ufuc_data_directory=Path(ufuc_data_directory),
            category_limit=category_limit,
            num_reference_images=num_reference_images,
        )

        print(
            f"[Eval] Test profile loaded from {test_profile_directory} ({len(profile_info.tests)} tests)"
        )
    else:
        print(
            f"[Eval] Test profile loaded from {test_profile_directory} ({len(profile_info.tests)} tests)"
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

    test_results = TestResults(directory=Path(test_results_dir))
    test_results.save()

    total_tests = len(profile_info.tests)

    for test_idx, test_info in enumerate(profile_info.tests):
        print()
        print(f"[Eval] {total_tests - test_idx} tests remaining")

        test_output_dir = f"{test_results_dir}/{test_info.name()}"

        print(
            f"[Eval] Test {test_info.name()} begins (Saving to {test_output_dir}):"
        )
        print()

        seed = test_info.seed
        test_performance = FontMetrics(device=args.device)

        os.makedirs(f"{test_output_dir}", exist_ok=True)

        total_instances = len(test_info.test_instances)

        runtime_list = []

        for inst_idx, test_instance in enumerate(test_info.test_instances):
            print(
                f"[{test_idx + 1}/{total_tests}][{inst_idx + 1}/{total_instances}] ", end=""
            )

            character = test_instance.character

            reference_images = [ReferenceImage.from_image_path(reference.path) for reference in test_instance.reference_images]

            out_image, runtime_secs = run_fontdiffuser_demo_mode(
                args=args,
                pipe=pipe,
                content_image=None,
                character=character,
                style_images=ReferenceImageList(reference_images, num_reference_images),
                ttf_path=ttf_path,
                k_shot=num_reference_images,
                seed=seed,
            )

            assert out_image is not None


            actual_image = Image.open(test_instance.actual_image.path).convert("RGB")
            if out_image.size != actual_image.size:
                actual_image = actual_image.resize(
                    out_image.size, Image.Resampling.BILINEAR
                )

            out_image.save(f"{test_output_dir}/{test_instance.actual_image.path.stem}.png")
            actual_image.save(f"{test_output_dir}/{test_instance.actual_image.path.stem}_actual.png")

            output_image_batch = torch.stack([toTensor(out_image)])
            actual_image_batch = torch.stack([toTensor(actual_image)])
            runtime_list.append(runtime_secs)

            test_performance.update(output_image_batch, actual_image_batch)


        test_performance_result = test_performance.compute()
        test_result = TestResult(
            data_category=test_info.data_category,
            font_category=test_info.font_category,
            fid=test_performance_result["fid"],
            ssim=test_performance_result["ssim"],
            lpips=test_performance_result["lpips"],
            l1=test_performance_result["l1"],
            runtime_secs=runtime_list,
        )
        test_results.add(test_result)
        test_results.save()

        print()
        print(
            f"[Eval] Sampling and evaluation for test {test_info.name()} ended in {test_result.total_runtime}s"
        )

    test_results.analyze()
    test_results.save()

    print()
    print("[Eval] Evaluation finished")
    print(f"Saved to {test_results_dir}")
    print(f"Total time: {sum(test_result.total_runtime for test_result in test_results.test_results)} seconds")


if __name__ == "__main__":
    main()
