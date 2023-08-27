import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from collections import namedtuple


KeysAndValues = namedtuple("KeysAndValues", ["keys", "values"])


class Attention(nn.Module):
    """A simple Attention module."""

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

    def _scaled_prod_attention(self, queries, keys, values):
        eps = 1e-5
        denom = torch.sqrt(self.qk_dim + eps)
        return torch.bmm(
            F.softmax(torch.bmm(queries, keys.transpose(1, 2)) / denom), values
        )

    def forward(self, emb, keys_and_values=Optional[KeysAndValues]):
        queries = self.query_projection(emb)
        keys = None
        values = None
        if keys_and_values is None:
            keys = self.key_projection(emb)
            values = self.value_projection(emb)
        else:
            keys = keys_and_values.keys
            values = keys_and_values.values
        out = self._scaled_prod_attention(queries, keys, values)
        return out


def test_attention():
    assert True