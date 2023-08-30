import torch
import torch.nn as nn
from torch import Tensor

from dlpr_code.constants import SMOOTHING_TERM


class TemporalBatchNormalization(nn.Module):
    """A simple module for performing Temporal Batch Normalization. Applicable
    to 1D inputs.

    Attributes
    ----------
    input_dim : int
        The dimension of the inputs representing the number of elements for
        which Batch Normalization will be performed.
    scale : torch.Tensor
        Learned scale parameter, applied to all positions.
    shift : torch.Tensor
        Learned shift parameter, applied to all positions.
    """

    def __init__(self, input_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.scale = nn.Linear(input_dim, 1)
        self.shift = nn.Linear(input_dim, 1)

    def forward(self, x_batch: Tensor) -> Tensor:
        """Forward pass for TemporalBatchNormalization.

        Parameters
        ----------
        x_batch : Tensor
            Input tensor. Should represent the outputs of an affine
            transformation, just before passing to activations operation.
            Dimension [B x input_dim], where B is batch size.

        Returns
        -------
        Tensor
            Normalized inputs ready for activation operation.
        """

        # estimate the means and std_dev
        mu_x = torch.mean(x_batch, dim=0)  # mean over batch
        sigma_x = torch.std(x_batch, dim=0, correction=0)  # biased std dev
        z_batch = (x_batch - mu_x) / (sigma_x + SMOOTHING_TERM)

        # apply shift and scale using learned (affine) parameters
        scale = self.scale(x_batch)
        shift = self.shift(x_batch)
        tilde_z_batch = scale * z_batch + shift
        return tilde_z_batch


class SpatialBatchNormalization(nn.Module):
    """A simple module for performing Temporal Batch Normalization. Applicable
    to 2D inputs. e.g., Images

    Attributes
    ----------
    input_dim : int
        The number of channels the image input consists of.
    """

    def __init__(self, num_channels: int):
        super().__init__()
        self.num_channels = num_channels

    def forward(self, x_batch: Tensor) -> Tensor:
        """Forward pass for TemporalBatchNormalization.

        Parameters
        ----------
        x_batch : Tensor
            Input tensor. Should represent the outputs of an affine
            transformation, just before passing to activations operation.
            Dimension [B x input_dim], where B is batch size.

        Returns
        -------
        Tensor
            Normalized inputs ready for activation operation.
        """

        # estimate the means and std_dev for each channel
        z_batch = torch.empty(x_batch.shape)
        for c in self.num_channels:
            mu_x = torch.mean(x_batch[:, c], dim=0)  # mean over batch
            sigma_x = torch.std(
                x_batch[:, c], dim=0, correction=0
            )  # biased std dev
            z_batch[:, c] = (x_batch[:, c] - mu_x) / (sigma_x + SMOOTHING_TERM)

        return z_batch
