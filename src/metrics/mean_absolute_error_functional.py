# This script is provided by the FYP24 project group.
# This script supports the Mean Absolute Error metric on mean_absolute_error.py.

# This script is adapted from the original Mean Squared Error script provided by the TorchEval authors.
# The original script can be found at
# https://github.com/pytorch/torcheval/blob/2c7dfb3768335ad7438bfb9bbe5c050e3f0780dc/torcheval/metrics/functional/regression/mean_squared_error.py

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from typing import Optional, Tuple

import torch


@torch.inference_mode()
def mean_absolute_error(
    input: torch.Tensor,
    target: torch.Tensor,
    *,
    sample_weight: Optional[torch.Tensor] = None,
    multioutput: str = "uniform_average",
) -> torch.Tensor:
    """
    Compute Mean Absolute Error, which is the mean of absolute error of `input` and `target`
    Its class version is ``torcheval.metrics.MeanAbsoluteError``.

    Args:
        input (Tensor): Tensor of predicted values with shape of (n_sample, n_output).
        target (Tensor): Tensor of ground truth values with shape of (n_sample, n_output).
        sample_weight (Optional):
            Tensor of sample weights with shape of (n_sample, ). Defaults to None.
        multioutput (Optional):
            - ``'uniform_average'`` [default]:
                Return scores of all outputs are averaged with uniform weight.
            - ``'raw_values'``:
                Return a full set of scores.
    Raises:
        ValueError:
            - If value of multioutput does not exist in (``raw_values``, ``uniform_average``).
            - If the dimension of `input` or `target` is not 1D or 2D.
            - If the `input` and `target` do not have the same size.
            - If the first dimension of `input`, `target` and `sample_weight` are not the same.

    Examples::

        >>> import torch
        >>> input = torch.tensor([0.9, 0.5, 0.3, 0.5])
        >>> target = torch.tensor([0.5, 0.8, 0.2, 0.8])
        >>> mean_absolute_error(input, target)
        tensor(0.2750)

        >>> input = torch.tensor([[0.9, 0.5], [0.3, 0.5]])
        >>> target = torch.tensor([[0.5, 0.8], [0.2, 0.8]])
        >>> mean_absolute_error(input, target)
        tensor(0.2750)

        >>> input = torch.tensor([[0.9, 0.5], [0.3, 0.5]])
        >>> target = torch.tensor([[0.5, 0.8], [0.2, 0.8]])
        >>> mean_absolute_error(input, target, multioutput="raw_values")
        tensor([0.2500, 0.3000])

        >>> input = torch.tensor([[0.9, 0.5], [0.3, 0.5]])
        >>> target = torch.tensor([[0.5, 0.8], [0.2, 0.8]])
        >>> mean_absolute_error(input, target, sample_weight=torch.tensor([0.2, 0.8]))
        tensor(0.2300)
    """
    _mean_absolute_error_param_check(multioutput)
    sum_absolute_error, sum_weight = _mean_absolute_error_update(
        input, target, sample_weight
    )
    return _mean_absolute_error_compute(sum_absolute_error, multioutput, sum_weight)


def _mean_absolute_error_update(
    input: torch.Tensor,
    target: torch.Tensor,
    sample_weight: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    _mean_absolute_error_update_input_check(input, target, sample_weight)
    return _update(input, target, sample_weight)


@torch.jit.script
def _update(
    input: torch.Tensor,
    target: torch.Tensor,
    sample_weight: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    absolute_error = torch.abs(target - input)
    if sample_weight is None:
        sum_absolute_error = absolute_error.sum(dim=0)
        # When input sample_weight is None, weight defaults to 1.0.
        sum_weight = torch.tensor(target.size(0), device=target.device)
    else:
        if absolute_error.ndim == 2:
            sample_weight = sample_weight.unsqueeze(-1)
        sum_absolute_error = (absolute_error * sample_weight).sum(dim=0)
        sum_weight = sample_weight.sum(dim=0).squeeze()
    return sum_absolute_error, sum_weight


def _mean_absolute_error_compute(
    sum_absolute_error: torch.Tensor,
    multioutput: str,
    sum_weight: torch.Tensor,
) -> torch.Tensor:
    eps = torch.finfo(torch.float64).eps
    sign = sum_weight.sign()
    raw_values = sum_absolute_error / (sum_weight.abs().clamp(min=eps) * sign)
    if multioutput == "raw_values":
        return raw_values
    else:
        return raw_values.mean()


def _mean_absolute_error_update_input_check(
    input: torch.Tensor,
    target: torch.Tensor,
    sample_weight: Optional[torch.Tensor],
) -> None:
    if input.ndim >= 3 or target.ndim >= 3:
        raise ValueError(
            "The dimension `input` and `target` should be 1D or 2D, "
            f"got shapes {input.shape} and {target.shape}."
        )
    if input.size() != target.size():
        raise ValueError(
            "The `input` and `target` should have the same size, "
            f"got shapes {input.shape} and {target.shape}."
        )
    if isinstance(sample_weight, torch.Tensor) and target.size(0) != sample_weight.size(
        0
    ):
        raise ValueError(
            "The first dimension of `input`, `target` and `sample_weight` should be the same size, "
            f"got shapes {input.shape}, {target.shape} and {sample_weight.shape}."
        )


def _mean_absolute_error_param_check(multioutput: str) -> None:
    if multioutput not in ("raw_values", "uniform_average"):
        raise ValueError(
            "The `multioutput` must be either `raw_values` or `uniform_average`, "
            f"got multioutput={multioutput}."
        )
