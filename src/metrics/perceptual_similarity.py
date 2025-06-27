# This script is provided by the FYP24 project group.
# This script is an implementation of LPIPS using TorchEval Metric.

from lpips import LPIPS
from typing import Iterable, Optional, TypeVar
import torch
from torcheval.metrics.metric import Metric

TPerceptualSimilarity = TypeVar("TPerceptualSimilarity")


class PerceptualSimilarity(Metric[torch.Tensor]):

    def __init__(
        self: TPerceptualSimilarity, device: Optional[torch.device] = None
    ) -> None:
        super().__init__(device=device)
        self.lpips = LPIPS().to(device)

        # Initialize state variables used to compute LPIPS
        self._add_state("sum_diff", torch.tensor(0.0, device=device))
        self._add_state("sum_weights", torch.tensor(0.0, device=device))

    @torch.inference_mode()
    def update(
        self: TPerceptualSimilarity, images_1: torch.Tensor, images_2: torch.Tensor
    ) -> TPerceptualSimilarity:
        """
        Update the metric with the new input and target.

        Args:
            images_1 (torch.Tensor): The predicted tensors (size: Nx3xWxH).
            images_2 (torch.Tensor): The target tensors (size: Nx3xWxH).
        """
        self._LPIPS_update_input_check(images_1)
        self._LPIPS_update_input_check(images_2)

        images_1 = images_1.to(self.device)
        images_2 = images_2.to(self.device)

        diff = self.lpips(images_1, images_2)
        self.sum_diff += torch.sum(diff)
        self.sum_weights += diff.numel()

        return self

    @torch.inference_mode()
    def compute(self: TPerceptualSimilarity) -> torch.Tensor:
        """
        Compute the metric based on the accumulated inputs and targets.

        Returns:
            torch.Tensor: The computed metric.
        """
        sum_diff = self.sum_diff
        sum_weight = self.sum_weights

        eps = torch.finfo(torch.float64).eps
        sign = sum_weight.sign()
        raw_values = sum_diff / (sum_weight.abs().clamp(min=eps) * sign)

        return raw_values.mean()

    @torch.inference_mode()
    def merge_state(
        self: TPerceptualSimilarity, metrics: Iterable[TPerceptualSimilarity]
    ) -> TPerceptualSimilarity:
        """
        Merge the state of another LPIPS instance into this instance.

        Args:
            metrics (Iterable[LPIPS]): The other LPIPS instance(s) whose state will be merged into this instance.
        """
        for metric in metrics:
            self.sum_diff += metric.sum_diff.to(self.device)
            self.sum_weights += metric.sum_weights.to(self.device)

        return self

    def _LPIPS_update_input_check(
        self: TPerceptualSimilarity, images: torch.Tensor
    ) -> None:
        if not torch.is_tensor(images):
            raise ValueError(f"Expected tensor as input, but got {type(images)}.")

        if images.dim() != 4:
            raise ValueError(
                f"Expected 4D tensor as input. But input has {images.dim()} dimenstions."
            )

        if images.size()[1] != 3:
            raise ValueError(f"Expected 3 channels as input. Got {images.size()[1]}.")

        if images.min() < 0 or images.max() > 1:
            raise ValueError(
                "In LPIPS, images are expected to be in the [0, 1] interval"
            )
