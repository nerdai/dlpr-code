from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from dlpr_code.constants import SMOOTHING_TERM


class Attention(nn.Module):
    """A simple Attention module. This class only takes one input sequence at
    at time.

    Attributes
    ----------
    model_dim : int
        Dimension of the model.
    qk_dim : int
        Dimension of embeddings for queries and keys.
    v_dim : Optional[int]
        Dimension of embeddings for values.
    query_projection : torch.Tensor
        Linear projection for queries.
    key_projection : torch.Tensor
        Linear projection for keys.
    value_projection : torch.Tensor
        Linear projection for values.
    """

    def __init__(self, model_dim: int, qk_dim: int, v_dim: Optional[int]):
        super().__init__()
        self.model_dim = model_dim
        self.qk_dim = qk_dim
        self.v_dim = v_dim
        self.query_projection = nn.Linear(model_dim, qk_dim)
        self.key_projection = nn.Linear(model_dim, qk_dim)
        if v_dim is None:
            self.value_projection = nn.Linear(model_dim, qk_dim)
        else:
            self.value_projection = nn.Linear(model_dim, v_dim)

    def __scaled_prod_attention(
        self, *, queries: Tensor, keys: Tensor, values: Tensor
    ) -> Tensor:
        """Helper function used in forward pass. One of the ways to compute
        attention, namely via scaled dot product.

        Parameters
        ----------
        queries : torch.Tensor
            A tensor containining query representations.
        keys : torch.Tensor
            A tensor containining keys representations.
        values : torch.Tensor
            A tensor containining values representations.

        Returns
        -------
        _type_
            _description_
        """
        denom = torch.sqrt(self.qk_dim + SMOOTHING_TERM)
        return torch.mm(
            F.softmax(
                torch.mm(queries, keys.transpose(1, 2)) / denom,
                dim=-1,  # along the last dimension post
            ),
            values,
        )

    def forward(
        self, seq: Tensor, keys: Optional[Tensor], values: Optional[Tensor]
    ) -> Tensor:
        """Forward pass for Attention block. Uses scaled-dot product attention.

        Parameters
        ----------
        seq : Tensor
            Sequence of embeddings (model_dim) or hidden state input to
            the Attention block.
        keys : Optional[Tensor]
            Optional keys vector. If not passed, then computes self-attention.
        values : Optional[Tensor]
            Optional keys vector. If not passed, then computes self-attention.

        Returns
        -------
        Tensor
            Returns the provided embeddings after performing Attention.
        """
        queries = self.query_projection(seq)
        if keys is None:
            keys = self.key_projection(seq)
        if values is None:
            values = self.value_projection(seq)
        out = self.__scaled_prod_attention(
            queries=queries, keys=keys, values=values
        )
        return out
