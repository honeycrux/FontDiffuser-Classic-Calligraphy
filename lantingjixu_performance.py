from pathlib import Path
import torch
from torcheval.metrics import StructuralSimilarity, FrechetInceptionDistance
from src.metrics.mean_absolute_error import MeanAbsoluteError
from src.metrics.perceptual_similarity import PerceptualSimilarity
import torchvision.transforms as TF
from PIL import Image

IMAGE_EXTENSIONS = {"bmp", "jpg", "jpeg", "pgm", "png", "ppm", "tif", "tiff", "webp"}

if __name__ == '__main__':
    comparison_dataset_dir = 'outputs/original'
    ground_truth_dataset_dir = 'outputs/target'

    device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")

    fid_metric = FrechetInceptionDistance(device=device)
    ssim_metric = StructuralSimilarity(device=device)
    l1_metric = MeanAbsoluteError(device=device)
    lpips_metric = PerceptualSimilarity(device=device)

    comparison_dataset_dir_path = Path(comparison_dataset_dir)
    ground_truth_dataset_dir_path = Path(ground_truth_dataset_dir)

    toTensor = TF.ToTensor()

    for comparison_file in comparison_dataset_dir_path.iterdir():
        if comparison_file.suffix.lower().lstrip('.') in IMAGE_EXTENSIONS:
            target_file = ground_truth_dataset_dir_path.joinpath(comparison_file.name)
            if not target_file.exists():
                print(f'File {comparison_file.name} does not exist in the ground truth dataset, skipping')
                break

            # Load images
            ground_truth_image = Image.open(target_file).convert('RGB')
            comparison_image = Image.open(comparison_file).convert('RGB')
            if comparison_image.size != ground_truth_image.size:
                comparison_image = comparison_image.resize(ground_truth_image.size)

            # Convert to tensor
            ground_truth_image = toTensor(ground_truth_image).to(device)
            comparison_image = toTensor(comparison_image).to(device)

            # Add batch dimension
            ground_truth_image_batch = ground_truth_image.unsqueeze(dim=0)
            comparison_image_batch = comparison_image.unsqueeze(dim=0)

            # Update metrics
            fid_metric.update(comparison_image_batch, is_real=False)
            fid_metric.update(ground_truth_image_batch, is_real=True)
            ssim_metric.update(comparison_image_batch, ground_truth_image_batch)
            lpips_metric.update(comparison_image_batch, ground_truth_image_batch)

            for i in range(ground_truth_image.shape[0]):
                l1_metric.update(comparison_image[i], ground_truth_image[i])

    fid_value = fid_metric.compute()
    print(f'FID value: {fid_value}')

    ssim_value = ssim_metric.compute()
    print(f'SSIM value: {ssim_value}')

    lpips_value = lpips_metric.compute()
    print(f'LPIPS value: {lpips_value}')

    l1_value = l1_metric.compute()
    print(f'L1 value: {l1_value}')
