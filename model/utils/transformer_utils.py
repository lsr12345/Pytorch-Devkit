'''
# Author: Shaoran Lu
# Date: 2021/10/04
# Email: lushaoran92@gmail.com
# Description: 

example:

'''

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from torch.nn.parameter import Parameter

import numpy as np
import math

import os
import sys
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.append(os.path.abspath(os.path.join(__dir__, '../')))

from model.utils.ops import clones

class MultiHeadAttention(nn.Module):
    def __init__(self, multi_attention_heads, dimensions, dropout=0.1):
        """

        :param _multi_attention_heads: number of self attention head
        :param _dimensions: dimension of model
        :param _dropout:
        """
        super(MultiHeadAttention, self).__init__()

        assert dimensions % multi_attention_heads == 0
        self.d_k = int(dimensions / multi_attention_heads)
        self.h = multi_attention_heads
        self.linears = clones(nn.Linear(dimensions, dimensions), 4)
        self.attention = None
        self.dropout = nn.Dropout(p=dropout)

    def dot_product_attention(self, query, key, value, mask):
        """
        Compute 'Scaled Dot Product Attention

        :param _query: (N, h, seq_len, d_q), h is multi-head
        :param _key: (N, h, seq_len, d_k)
        :param _value: (N, h, seq_len, d_v)
        :param _mask: None or (N, 1, seq_len, seq_len), 0 will be replaced with -1e9
        :return:
        """

        d_k = value.size(-1)
        score = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            score = score.masked_fill(mask == 0, -1e9)
        p_attn = F.softmax(score, dim=-1)
        return torch.matmul(p_attn, value), p_attn

    def forward(self, query, key, value, mask):
        batch_size = query.size(0)

        query, key, value = \
            [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        product_and_attention = self.dot_product_attention(query, key, value, mask=mask)
        x = product_and_attention[0]

        x = x.transpose(1, 2).contiguous() \
            .view(batch_size, -1, self.h * self.d_k)

        return self.linears[-1](x)

class FeedForwarding(nn.Module):
    def __init__(self, _dimensions, _feed_forward_dimensions, _dropout=0.1):
        super(FeedForwarding, self).__init__()
        self.w_1 = nn.Linear(_dimensions, _feed_forward_dimensions)
        self.w_2 = nn.Linear(_feed_forward_dimensions, _dimensions)
        self.dropout = nn.Dropout(p=_dropout)

    def forward(self, _input_tensor):
        return self.w_2(self.dropout(F.relu(self.w_1(_input_tensor))))

class PositionalEncoding(nn.Module):
    def __init__(self, emb_size, dropout=0.1, maxlen=5000):
        super(PositionalEncoding, self).__init__()


        den = torch.exp(- torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        self.pos_embedding = pos_embedding.unsqueeze(0)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x): 
        """Forward pass.
        Args:
            x: (B, len, d_model)
        Returns:
            (B, len, d_model)
        """
        return self.dropout(x + self.pos_embedding[:,  :x.size(1),  :].to(x.device))

class PositionalEncoding2D(nn.Module):
    def __init__(self, emb_size, dropout=0.1, max_h=1000, max_w=1000):
        super(PositionalEncoding2D, self).__init__()


        self.emb_size = emb_size
        assert emb_size % 2 == 0, f"Embedding depth {emb_size} is not even"
        pe_h = self.make_pe(emb_size // 2, maxlen=max_h)
        pe_w = self.make_pe(emb_size // 2, maxlen=max_w)

        pe_h = pe_h.permute(2, 1, 0).expand(-1, -1, max_w)
        pe_w = pe_w.permute(2, 0, 1).expand(-1, max_h, -1)

        pe = torch.cat([pe_h, pe_w], dim=0)
        self.pe = pe.unsqueeze(0)
        self.dropout = nn.Dropout(dropout)

    def make_pe(self, emb_size, maxlen=2000):
        den = torch.exp(- torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(0)
        return pos_embedding

    def forward(self, x):
        """Forward pass.
        Args:
            x: (B, d_model, H, W)
        Returns:
            (B, d_model, H, W)
        """
        assert x.shape[1] == self.pe.shape[1]
        return self.dropout(x + self.pe[:, :, : x.size(2), : x.size(3)].to(x.device))

class TransformerDecoderLayer(nn.Module):
    def __init__(self, nhead, d_model, n_layers, dropout, dim_feedforward, n_classes, PAD_IDX=1):
        
        super(TransformerDecoderLayer, self).__init__()
        self.attention = MultiHeadAttention(nhead, d_model, dropout)
        self.source_attention = MultiHeadAttention(nhead, d_model, dropout)
        self.position_feed_forward = FeedForwarding(d_model, dim_feedforward, dropout)
        self.position = PositionalEncoding(d_model, dropout)
        self.stacks = n_layers
        self.dropout = torch.nn.Dropout(dropout)
        self.layer_norm = torch.nn.LayerNorm(d_model, eps=1e-6)
        self.embedding = nn.Embedding(n_classes, d_model)
        self.sqrt_model_size = math.sqrt(d_model)
        self.padding_symbol = PAD_IDX

    def generate_target_mask(self, source, target):
        target_pad_mask = (target != self.padding_symbol).unsqueeze(1).unsqueeze(3)
        target_length = target.size(1)
        target_sub_mask = torch.tril(
            torch.ones((target_length, target_length), dtype=torch.uint8, device=source.device)
        )
        source_mask = torch.ones((target_length, source.size(1)), dtype=torch.uint8, device=source.device)
        target_mask = target_pad_mask & target_sub_mask.bool()
        return source_mask, target_mask

    def eval(self):
        self.attention.eval()
        self.source_attention.eval()
        self.position_feed_forward.eval()
        self.position.eval()
        self.dropout.eval()
        self.layer_norm.eval()
        self.embedding.eval()

    def forward(self, target_result, memory):
        target = self.embedding(target_result) * self.sqrt_model_size
        target = self.position(target)

        if self.padding_symbol is None:
            source_mask, target_mask = None, None
        else:
            source_mask, target_mask = self.generate_target_mask(memory, target_result)
        output = target
        for i in range(self.stacks):
            normed_output = self.layer_norm(output)
            output = output + self.dropout(
                self.attention(normed_output, normed_output, normed_output, target_mask)
            )
            normed_output = self.layer_norm(output)
            output = output + self.dropout(self.source_attention(normed_output, memory, memory, source_mask))
            normed_output = self.layer_norm(output)
            output = output + self.dropout(self.position_feed_forward(normed_output))
        return self.layer_norm(output)