import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, max_seq_len=1024, dropout=0.0):
        super().__init__()
        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

class MLP(nn.Module):
    pass

class TransformerBlock(nn.Module):
    pass