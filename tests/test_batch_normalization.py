from unittest import TestCase

import numpy as np
import torch

from dlpr_code.batch_normalization import (
    SpatialBatchNormalization,
    TemporalBatchNormalization,
)
from dlpr_code.constants import SMOOTHING_TERM


class BatchNormalizationTest(TestCase):
    def test_temporal_bn(self):
        model_dim = 10
        batch_size = 5
        temporal_bn = TemporalBatchNormalization(input_dim=model_dim)
        sample = torch.kron(
            torch.arange(batch_size).unsqueeze(1), torch.ones(model_dim)
        )
        results = temporal_bn(sample)

        # expected
        batch_mean = 2.0
        batch_std = np.sqrt(2.0)
        expected = torch.kron(
            (torch.arange(batch_size).unsqueeze(1) - batch_mean)
            / (batch_std + SMOOTHING_TERM),
            torch.ones(model_dim),
        )
        self.assertAlmostEqual(
            torch.norm(results - expected),
            0.0,
            None,
            "result not almost equal to expected",
            1e-5,
        )

    def test_spatial_bn(self):
        height = 4
        width = 4
        channels = 3
        spatial_bn = SpatialBatchNormalization(num_channels=channels)
        sample = torch.cat(
            (
                torch.zeros(channels, height, width).unsqueeze(0),
                torch.ones(channels, height, width).unsqueeze(0),
                torch.ones(channels, height, width).unsqueeze(0) * 2,
                torch.ones(channels, height, width).unsqueeze(0) * 3,
                torch.ones(channels, height, width).unsqueeze(0) * 4,
            )
        )
        results = spatial_bn(sample)

        # expected
        batch_mean = 2.0
        batch_std = np.sqrt(2.0)
        expected = torch.cat(
            (
                (
                    torch.zeros(channels, height, width).unsqueeze(0)
                    - batch_mean
                )
                / (batch_std + SMOOTHING_TERM),
                (torch.ones(channels, height, width).unsqueeze(0) - batch_mean)
                / (batch_std + SMOOTHING_TERM),
                (
                    torch.ones(channels, height, width).unsqueeze(0) * 2
                    - batch_mean
                )
                / (batch_std + SMOOTHING_TERM),
                (
                    torch.ones(channels, height, width).unsqueeze(0) * 3
                    - batch_mean
                )
                / (batch_std + SMOOTHING_TERM),
                (
                    torch.ones(channels, height, width).unsqueeze(0) * 4
                    - batch_mean
                )
                / (batch_std + SMOOTHING_TERM),
            )
        )

        self.assertAlmostEqual(
            torch.norm(results - expected),
            0.0,
            None,
            "result not almost equal to expected",
            1e-5,
        )
