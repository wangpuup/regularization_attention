#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Shigeki Karita
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
# Pu Wang: 2025, added DropHead (W. Zhou, T. Ge, F. Wei, M. Zhou, and K. Xu, “Scheduled DropHead: A regularization method for transformer models,” in EMLP 2020 Findings.
# and DropAttention (Z. Lin, P. Liu, L. Huang, J. Chen, X. Qiu, and X. Huang, “Dropattention: A regularization method for fully-connected self-attention networks.”

"""Multi-Head Attention layer definition."""

import math
import random

import torch
from torch import nn
import torch.nn.functional as F

class MultiHeadedAttention(nn.Module):
    """Multi-Head Attention layer.

    Args:
        n_head (int): The number of heads.
        n_feat (int): The number of features.
        dropout_rate (float): Dropout rate.

    """

    def __init__(self, n_head, n_feat, n_attn, dropout_rate, drop_attention_mode, drop_attention_rate, drop_attention_window):
        """Construct an MultiHeadedAttention object."""
        super(MultiHeadedAttention, self).__init__()
        assert n_feat % n_head == 0
        # We assume d_v always equals d_k
        assert n_attn % n_head == 0
        self.d_k = n_attn // n_head
        self.h = n_head
        self.linear_q = nn.Linear(n_feat, n_attn)
        self.linear_k = nn.Linear(n_feat, n_attn)
        self.linear_v = nn.Linear(n_feat, n_attn)
        self.linear_out = nn.Linear(n_attn, n_feat)
        
        self.attn = None
        self.dropout = nn.Dropout(p=dropout_rate)
        self.drop_attention_mode = drop_attention_mode
        self.drop_attention_rate = drop_attention_rate
        self.drop_attention_window = drop_attention_window

    def set_drophead_rate(self, new_rate):
        self.drop_attention_rate = new_rate

    def forward_qkv(self, query, key, value):
        """Transform query, key and value.

        Args:
            query (torch.Tensor): Query tensor (#batch, time1, size).
            key (torch.Tensor): Key tensor (#batch, time2, size).
            value (torch.Tensor): Value tensor (#batch, time2, size).

        Returns:
            torch.Tensor: Transformed query tensor (#batch, n_head, time1, d_k).
            torch.Tensor: Transformed key tensor (#batch, n_head, time2, d_k).
            torch.Tensor: Transformed value tensor (#batch, n_head, time2, d_k).

        """
        n_batch = query.size(0)
        q = self.linear_q(query).view(n_batch, -1, self.h, self.d_k)
        k = self.linear_k(key).view(n_batch, -1, self.h, self.d_k)
        v = self.linear_v(value).view(n_batch, -1, self.h, self.d_k)
        q = q.transpose(1, 2)  # (batch, head, time1, d_k)
        k = k.transpose(1, 2)  # (batch, head, time2, d_k)
        v = v.transpose(1, 2)  # (batch, head, time2, d_k)

        return q, k, v

    def forward_attention(self, value, scores, mask):
        """Compute attention context vector.

        Args:
            value (torch.Tensor): Transformed value (#batch, n_head, time2, d_k).
            scores (torch.Tensor): Attention score (#batch, n_head, time1, time2).
            mask (torch.Tensor): Mask (#batch, 1, time2) or (#batch, time1, time2).
            self.drop_attention_mode (str): Drop attention mode.
            - 'h': DropHead
            - 'e': DropAttention (element-wise)
            - 'c': DropAttention (column-wise)
            - 'None': regular dropout

        Returns:
            torch.Tensor: Transformed value (#batch, time1, d_model)
                weighted by the attention score (#batch, time1, time2).

        """
        n_batch = value.size(0)
        if mask is not None:
            mask = mask.unsqueeze(1).eq(0)  # (batch, 1, *, time2)
            min_value = torch.finfo(scores.dtype).min
            scores = scores.masked_fill(mask, min_value)
            self.attn = torch.softmax(scores, dim=-1).masked_fill(
                mask, 0.0
            )  # (batch, head, time1, time2)
        else:
            self.attn = torch.softmax(scores, dim=-1)  # (batch, head, time1, time2)

        if self.drop_attention_mode == "h":

            x = torch.matmul(self.attn, value)  # (batch, head, time1, d_k)
            
            if self.training:

                B, H, T1, T2 = x.shape
                device = x.device

                mask_head = torch.ones(B, H, 1, 1, device=device)
                bern_prob = 1.0 - self.drop_attention_rate
                mask_head = torch.bernoulli(mask_head * bern_prob)

                mask_head_sum = mask_head.sum(dim=1, keepdim=True)  # (B, 1, 1, 1)
                mask_head = torch.where(mask_head_sum == 0, torch.ones_like(mask_head), mask_head)

                x_masked = x * mask_head
                norm_head = mask_head.sum(dim=1, keepdim=True) + 1e-8  # Avoid division by zero
                x_masked = x_masked * (self.h / norm_head)
                x = x_masked
            
            x = (
                x.transpose(1, 2).contiguous().view(n_batch, -1, self.h * self.d_k)
            )  # (batch, time1, d_model)

            return self.linear_out(x)  # (batch, time1, d_model)
        
        elif self.drop_attention_mode in ["e", "c"]:

            if self.training:

                B, H, T1, T2 = self.attn.shape
                gamma = self.drop_attention_rate / self.drop_attention_window

                if self.drop_attention_mode == "e":
                    M = (torch.rand(B, H, T1, T2, device=self.attn.device) > gamma).float()
        
                elif self.drop_attention_mode == "c":
                    col_mask = (torch.rand(B, H, T2, device=self.attn.device) > gamma).float()
                    M = col_mask.unsqueeze(2).expand(B, H, T1, T2)  # (B, H, T1, T2)

                M_expanded = M.clone()

                for offset in range(1, self.drop_attention_window):
                    shifted = F.pad(M[..., :-offset], (offset, 0), value=1.0)
                    M_expanded *= shifted  # 0 anywhere in the window → 0

            else:
                M_expanded = torch.ones_like(self.attn)
            
            p_attn = self.attn * M_expanded
            # Renormalize (row-wise over time2)
            p_attn_sum = p_attn.sum(dim=-1, keepdim=True) + 1e-8
            p_attn = p_attn / p_attn_sum

            x = torch.matmul(p_attn, value)  # (batch, head, time1, d_k)
            x = (
                x.transpose(1, 2).contiguous().view(n_batch, -1, self.h * self.d_k)
            )  # (batch, time1, d_model)

            return self.linear_out(x)  # (batch, time1, d_model)
        
        elif self.drop_attention_mode == "None":
            # No drop attention, just return the standard attention output
            p_attn = self.dropout(self.attn)
            x = torch.matmul(p_attn, value)  # (batch, head, time1, d_k)
            x = (
                x.transpose(1, 2).contiguous().view(n_batch, -1, self.h * self.d_k)
            )  # (batch, time1, d_model)

            return self.linear_out(x)  # (batch, time1, d_model)

    def forward(self, query, key, value, mask):
        """Compute scaled dot product attention.

        Args:
            query (torch.Tensor): Query tensor (#batch, time1, size).
            key (torch.Tensor): Key tensor (#batch, time2, size).
            value (torch.Tensor): Value tensor (#batch, time2, size).
            mask (torch.Tensor): Mask tensor (#batch, 1, time2) or
                (#batch, time1, time2).

        Returns:
            torch.Tensor: Output tensor (#batch, time1, d_model).

        """
        q, k, v = self.forward_qkv(query, key, value)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        return self.forward_attention(v, scores, mask)

