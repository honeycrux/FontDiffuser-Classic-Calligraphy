# This script is provided by the FYP24 project group.
# This script calculates all of the metrics for a given batch of images.

from torcheval.metrics import StructuralSimilarity, FrechetInceptionDistance
from src.metrics.mean_absolute_error import MeanAbsoluteError
from src.metrics.perceptual_similarity import PerceptualSimilarity


class FontMetrics:
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
                self.l1_metric.update(
                    comparison_image_batch[i][j], ground_truth_image_batch[i][j]
                )

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
