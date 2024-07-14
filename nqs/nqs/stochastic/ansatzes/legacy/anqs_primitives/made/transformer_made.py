import torch as pt
from torch import nn

from typing import Tuple, Callable, Union

from nqs.base.constants import BASE_REAL_TYPE


class TransformerMADE(nn.Module):
    def __init__(self,
                 dim: int = None,
                 out_dim: int = None,
                 depth: int = None,
                 qubit_num: int = None,
                 head_num: int = None,
                 dtype=BASE_REAL_TYPE):
        super(TransformerMADE, self).__init__()
        self.dtype = dtype

        self.dim = dim
        self.out_dim = out_dim
        self.depth = depth
        self.qubit_num = qubit_num
        self.head_num = head_num

        self.pos_embedding = nn.Embedding(self.qubit_num + 1, dim, dtype=self.dtype)
        self.embedding = nn.Embedding(3, dim, dtype=self.dtype)

        transformer_layer = nn.TransformerEncoderLayer(d_model=self.dim,
                                                   nhead=self.head_num,
                                                   dim_feedforward=self.dim,
                                                   dropout=0.0,
                                                   batch_first=True,
                                                   dtype=self.dtype)
        self.transformer = nn.TransformerEncoder(encoder_layer=transformer_layer,
                                                 num_layers=self.depth,
                                                 enable_nested_tensor=False)
        self.decoder = nn.Linear(self.dim, self.out_dim, dtype=self.dtype)

    def forward(self, x: pt.Tensor):
        seq = pt.cat((pt.full((x.shape[0], 1), 2, dtype=pt.long, device=x.device), x), dim=-1)
        seq_len = seq.shape[-1]
        mask = pt.triu(pt.full((seq_len, seq_len), float('-inf'), dtype=self.dtype, device=seq.device), diagonal=1)
        seq = self.embedding(seq)
        pos_encoding = self.pos_embedding(pt.arange(seq_len, device=seq.device))
        seq = self.transformer(seq + pos_encoding, mask=mask, is_causal=True)

        return self.decoder(seq)
