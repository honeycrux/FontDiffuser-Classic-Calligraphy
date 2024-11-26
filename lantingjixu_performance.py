from pathlib import Path
import torch
from torcheval.metrics import StructuralSimilarity, FrechetInceptionDistance
from src.metrics.mean_absolute_error import MeanAbsoluteError
from src.metrics.perceptual_similarity import PerceptualSimilarity
import torchvision.transforms as TF
from PIL import Image

IMAGE_EXTENSIONS = {"bmp", "jpg", "jpeg", "pgm", "png", "ppm", "tif", "tiff", "webp"}

class LantingjixuPerformance:
    def __init__(self, device):
        self.device = device
        self.fid_metric = FrechetInceptionDistance(device=device)
        self.ssim_metric = StructuralSimilarity(device=device)
        self.lpips_metric = PerceptualSimilarity(device=device)
        self.l1_metric = MeanAbsoluteError(device=device)

    def update(self, comparison_image_batch, ground_truth_image_batch):
        if comparison_image_batch.device != self.device:
            comparison_image_batch = comparison_image_batch.to(self.device)
        if ground_truth_image_batch.device != self.device:
            ground_truth_image_batch = ground_truth_image_batch.to(self.device)

        self.fid_metric.update(comparison_image_batch, is_real=False)
        self.fid_metric.update(ground_truth_image_batch, is_real=True)
        self.ssim_metric.update(comparison_image_batch, ground_truth_image_batch)
        self.lpips_metric.update(comparison_image_batch, ground_truth_image_batch)

        for i in range(ground_truth_image_batch.shape[0]):
            for j in range(ground_truth_image_batch.shape[1]):
                self.l1_metric.update(comparison_image_batch[i][j], ground_truth_image_batch[i][j])

    def compute(self):
        fid_value = self.fid_metric.compute()
        ssim_value = self.ssim_metric.compute()
        lpips_value = self.lpips_metric.compute()
        l1_value = self.l1_metric.compute()

        return {
            "fid": fid_value.item(),
            "ssim": ssim_value.item(),
            "lpips": lpips_value.item(),
            "l1": l1_value.item(),
        }

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

    performance = LantingjixuPerformance(device=device)

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
            performance.update(comparison_image_batch, ground_truth_image_batch)

    perf = performance.compute()
    fid_value, ssim_value, lpips_value, l1_value = perf['fid'], perf['ssim'], perf['lpips'], perf['l1']

    print(f'FID value: {fid_value}')
    print(f'SSIM value: {ssim_value}')
    print(f'LPIPS value: {lpips_value}')
    print(f'L1 value: {l1_value}')
