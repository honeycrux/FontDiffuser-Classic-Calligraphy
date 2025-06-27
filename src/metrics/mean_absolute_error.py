# This script is provided by the FYP24 project group.
# This script is an implementation of L1 Loss (Mean Absolute Error) using TorchEval Metric.

# This script is adapted from the original Mean Squared Error script provided by the TorchEval authors.
# The original script can be found at
# https://github.com/pytorch/torcheval/blob/2c7dfb3768335ad7438bfb9bbe5c050e3f0780dc/torcheval/metrics/regression/mean_squared_error.py

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

# pyre-ignore-all-errors[16]: Undefined attribute of metric states.

from typing import Iterable, Optional, TypeVar

import torch

from .mean_absolute_error_functional import (
    _mean_absolute_error_compute,
    _mean_absolute_error_param_check,
    _mean_absolute_error_update,
)
from torcheval.metrics.metric import Metric

TMeanAbsoluteError = TypeVar("TMeanAbsoluteError")


class MeanAbsoluteError(Metric[torch.Tensor]):
    """
    Compute Mean Absolute Error, which is the mean of absolute error of `input` and `target`.
    Its functional version is :func:`torcheval.metrics.functional.mean_absolute_error`.

    Args:
        multioutput (str, Optional)
            - ``'uniform_average'`` [default]: Return scores of all outputs are averaged with uniform weight.
            - ``'raw_values'``: Return a full set of scores.
    Raises:
        ValueError:
            - If value of multioutput does not exist in (``raw_values``, ``uniform_average``).
            - If the dimension of `input` or `target` is not 1D or 2D.
            - If the `input` and `target` do not have the same size.
            - If the first dimension of `input`, `target` and `sample_weight` are not the same.

    Examples::

        >>> import torch
        >>> from torcheval.metrics import MeanAbsoluteError
        >>> metric = MeanAbsoluteError()
        >>> input = torch.tensor([0.9, 0.5, 0.3, 0.5])
        >>> target = torch.tensor([0.5, 0.8, 0.2, 0.8])
        >>> metric.update(input, target)
        >>> metric.compute()
        tensor(0.2750)

        >>> metric = MeanAbsoluteError()
        >>> input = torch.tensor([[0.9, 0.5], [0.3, 0.5]])
        >>> target = torch.tensor([[0.5, 0.8], [0.2, 0.8]])
        >>> metric.update(input, target)
        >>> metric.compute()
        tensor(0.2750)

        >>> metric = MeanAbsoluteError(multioutput="raw_values")
        >>> input = torch.tensor([[0.9, 0.5], [0.3, 0.5]])
        >>> target = torch.tensor([[0.5, 0.8], [0.2, 0.8]])
        >>> metric.update(input, target)
        >>> metric.compute()
        tensor([0.2500, 0.3000])

        >>> metric = MeanAbsoluteError()
        >>> input = torch.tensor([[0.9, 0.5], [0.3, 0.5]])
        >>> target = torch.tensor([[0.5, 0.8], [0.2, 0.8]])
        >>> metric.update(input, target, sample_weight=torch.tensor([0.2, 0.8]))
        >>> metric.compute()
        tensor(0.2300)
    """

    def __init__(
        self: TMeanAbsoluteError,
        *,
        multioutput: str = "uniform_average",
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__(device=device)
        _mean_absolute_error_param_check(multioutput)
        self.multioutput = multioutput
        self._add_state(
            "sum_absolute_error",
            torch.tensor(0.0, device=self.device),
        )
        self._add_state("sum_weight", torch.tensor(0.0, device=self.device))

    @torch.inference_mode()
    # pyre-ignore[14]: inconsistent override on *_:Any, **__:Any
    def update(
        self: TMeanAbsoluteError,
        input: torch.Tensor,
        target: torch.Tensor,
        *,
        sample_weight: Optional[torch.Tensor] = None,
    ) -> TMeanAbsoluteError:
        """
        Update states with the ground truth values and predictions.

        Args:
            input (Tensor): Tensor of predicted values with shape of (n_sample, n_output).
            target (Tensor): Tensor of ground truth values with shape of (n_sample, n_output).
            sample_weight (Optional):
                Tensor of sample weights with shape of (n_sample, ). Defaults to None.
        """
        (
            sum_absolute_error,
            sum_weight,
        ) = _mean_absolute_error_update(input, target, sample_weight)
        if self.sum_absolute_error.ndim == 0 and sum_absolute_error.ndim == 1:
            self.sum_absolute_error = sum_absolute_error
        else:
            self.sum_absolute_error += sum_absolute_error
        self.sum_weight += sum_weight
        return self

    @torch.inference_mode()
    def compute(self: TMeanAbsoluteError) -> torch.Tensor:
        """
        Return the Mean Absolute Error.

        NaN is returned if no calls to ``update()`` are made before ``compute()`` is called.
        """
        return _mean_absolute_error_compute(
            self.sum_absolute_error,
            self.multioutput,
            self.sum_weight,
        )

    @torch.inference_mode()
    def merge_state(
        self: TMeanAbsoluteError, metrics: Iterable[TMeanAbsoluteError]
    ) -> TMeanAbsoluteError:
        for metric in metrics:
            if (
                self.sum_absolute_error.ndim == 0
                and metric.sum_absolute_error.ndim == 1
            ):
                self.sum_absolute_error = metric.sum_absolute_error.to(self.device)
            else:
                self.sum_absolute_error += metric.sum_absolute_error.to(self.device)
            self.sum_weight += metric.sum_weight.to(self.device)
        return self
