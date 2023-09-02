from unittest import TestCase

import numpy as np
import torch

from dlpr_code.batch_normalization import TemporalBatchNormalization
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
