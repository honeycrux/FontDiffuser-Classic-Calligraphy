# This script is provided by the FYP24 project group.
# This is the driver code for a simple (now unused) implementation of performance evaluation for the LantingjiXu dataset.
# Given prepared original and target folders, the script will calculate the FID, SSIM, LPIPS, and L1 metrics.
# For our current evaluation process, check lantingjixu_eval.py instead, which also generates a test profile and runs sampling.

from pathlib import Path

import torch
import torchvision.transforms as TF
from PIL import Image

from src.metrics.font_metrics import FontMetrics

IMAGE_EXTENSIONS = {"bmp", "jpg", "jpeg", "pgm", "png", "ppm", "tif", "tiff", "webp"}


def main():
    comparison_dataset_dir = "outputs/original"
    ground_truth_dataset_dir = "outputs/target"

    device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")

    comparison_dataset_dir_path = Path(comparison_dataset_dir)
    ground_truth_dataset_dir_path = Path(ground_truth_dataset_dir)

    toTensor = TF.ToTensor()

    performance = FontMetrics(device=device)

    for comparison_file in comparison_dataset_dir_path.iterdir():
        if comparison_file.suffix.lower().lstrip(".") in IMAGE_EXTENSIONS:
            target_file = ground_truth_dataset_dir_path.joinpath(comparison_file.name)
            if not target_file.exists():
                print(
                    f"File {comparison_file.name} does not exist in the ground truth dataset, skipping"
                )
                break

            # Load images
            ground_truth_image = Image.open(target_file).convert("RGB")
            comparison_image = Image.open(comparison_file).convert("RGB")
            if comparison_image.size != ground_truth_image.size:
                comparison_image = comparison_image.resize(ground_truth_image.size)

            # Convert to tensor
            ground_truth_image = toTensor(ground_truth_image).to(device)
            comparison_image = toTensor(comparison_image).to(device)

            # Add batch dimension
            ground_truth_image_batch = ground_truth_image.unsqueeze(dim=0)
            comparison_image_batch = comparison_image.unsqueeze(dim=0)

            # Update metrics
            performance.update(comparison_image_batch, ground_truth_image_batch)

    perf = performance.compute()
    fid_value, ssim_value, lpips_value, l1_value = (
        perf["fid"],
        perf["ssim"],
        perf["lpips"],
        perf["l1"],
    )

    print(f"FID value: {fid_value}")
    print(f"SSIM value: {ssim_value}")
    print(f"LPIPS value: {lpips_value}")
    print(f"L1 value: {l1_value}")


if __name__ == "__main__":
    main()
