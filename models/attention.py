import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0

        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, T, C = x.size()

        q = self.q(x)
        k = self.k(x)
        v = self.v(x)

        q = q.view(B, T, self.num_heads, self.d_k).transpose(1,2)
        k = k.view(B, T, self.num_heads, self.d_k).transpose(1,2)
        v = v.view(B, T, self.num_heads, self.d_k).transpose(1,2)

        scores = torch.matmul(q, k.transpose(-2,-1)) / math.sqrt(self.d_k)
        attn = torch.softmax(scores, dim=-1)

        context = torch.matmul(attn, v)
        context = context.transpose(1,2).contiguous().view(B, T, C)

        return self.out(context)