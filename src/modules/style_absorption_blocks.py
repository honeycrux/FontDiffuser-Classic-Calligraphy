# This script is provided by the FYP24 project group.
# This script is adapted from attention.py, which is provided by authors of FontDiffuser.

from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn


class Transformer(nn.Module):
    r"""
    A stack of transformer blocks.
    """

    def __init__(self, encoder_blocks: nn.ModuleList, decoder_blocks: nn.ModuleList):
        super().__init__()
        self.encoder_blocks = encoder_blocks
        self.decoder_blocks = decoder_blocks

    def _set_attention_slice(self, slice_size):
        for block in self.encoder_blocks:
            assert block is torch.Module
            block._set_attention_slice(slice_size)
        for block in self.decoder_blocks:
            assert block is torch.Module
            block._set_attention_slice(slice_size)

    def forward(self, decoder_input, encoder_input=None):
        encoder_output = encoder_input
        for block in self.encoder_blocks:
            encoder_output = block(encoder_output)
        decoder_output = decoder_input
        for block in self.decoder_blocks:
            decoder_output = block(decoder_output, context=encoder_output)
        return decoder_output


class EncoderBlock(nn.Module):
    r"""
    A Encoder block.

    Parameters:
        query_dim (:obj:`int`): The size of the query vector.
        n_heads (:obj:`int`): The number of heads to use for multi-head attention.
        d_head (:obj:`int`): The number of channels in each head.
        dropout (:obj:`float`, *optional*, defaults to 0.0): The dropout probability to use.
        gated_ff (:obj:`bool`, *optional*, defaults to :obj:`False`): Whether to use a gated feed-forward network.
        checkpoint (:obj:`bool`, *optional*, defaults to :obj:`False`): Whether to use checkpointing.
    """

    def __init__(
        self,
        query_dim: int,
        n_heads: int,
        d_head: int,
        dropout=0.0,
        gated_ff: bool = True,
        ff_mult: int = 4,
        checkpoint: bool = True,
    ):
        super().__init__()
        self.self_attention = MultiHeadAttentionBlock(
            query_dim=query_dim, heads=n_heads, dim_head=d_head, dropout=dropout
        )
        self.self_attention_norm = nn.LayerNorm(query_dim)

        self.feed_forward = FeedForward(
            query_dim, dropout=dropout, glu=gated_ff, mult=ff_mult
        )
        self.feed_forward_norm = nn.LayerNorm(query_dim)

        self.checkpoint = checkpoint

    def _set_attention_slice(self, slice_size):
        self.self_attention._slice_size = slice_size

    def forward(self, hidden_states):
        hidden_states = (
            hidden_states.contiguous()
            if hidden_states.device.type == "mps"
            else hidden_states
        )
        hidden_states = (
            self.self_attention(self.self_attention_norm(hidden_states)) + hidden_states
        )
        hidden_states = (
            self.feed_forward(self.feed_forward_norm(hidden_states)) + hidden_states
        )
        return hidden_states


class DecoderBlock(nn.Module):
    r"""
    A Decoder block.

    Parameters:
        query_dim (:obj:`int`): The size of the query vector.
        n_heads (:obj:`int`): The number of heads to use for multi-head attention.
        d_head (:obj:`int`): The number of channels in each head.
        dropout (:obj:`float`, *optional*, defaults to 0.0): The dropout probability to use.
        context_dim (:obj:`int`, *optional*): The size of the context vector for cross attention.
        gated_ff (:obj:`bool`, *optional*, defaults to :obj:`False`): Whether to use a gated feed-forward network.
        checkpoint (:obj:`bool`, *optional*, defaults to :obj:`False`): Whether to use checkpointing.
    """

    def __init__(
        self,
        query_dim: int,
        n_heads: int,
        d_head: int,
        dropout=0.0,
        context_dim: Optional[int] = None,
        gated_ff: bool = True,
        ff_mult: int = 4,
        checkpoint: bool = True,
    ):
        super().__init__()
        self.self_attention = MultiHeadAttentionBlock(
            query_dim=query_dim, heads=n_heads, dim_head=d_head, dropout=dropout
        )
        self.self_attention_norm = nn.LayerNorm(query_dim)

        self.cross_attention = MultiHeadAttentionBlock(
            query_dim=query_dim,
            context_dim=context_dim,
            heads=n_heads,
            dim_head=d_head,
            dropout=dropout,
        )
        self.cross_attention_norm = nn.LayerNorm(query_dim)

        self.feed_forward = FeedForward(
            query_dim, dropout=dropout, glu=gated_ff, mult=ff_mult
        )
        self.feed_forward_norm = nn.LayerNorm(query_dim)

        self.checkpoint = checkpoint

    def _set_attention_slice(self, slice_size):
        self.self_attention._slice_size = slice_size
        self.cross_attention._slice_size = slice_size

    def forward(self, hidden_states, context=None):
        hidden_states = (
            hidden_states.contiguous()
            if hidden_states.device.type == "mps"
            else hidden_states
        )
        hidden_states = (
            self.self_attention(self.self_attention_norm(hidden_states)) + hidden_states
        )
        hidden_states = (
            self.cross_attention(
                self.cross_attention_norm(hidden_states), context=context
            )
            + hidden_states
        )
        hidden_states = (
            self.feed_forward(self.feed_forward_norm(hidden_states)) + hidden_states
        )
        return hidden_states


class FeedForward(nn.Module):
    r"""
    A feed-forward layer.

    Parameters:
        dim (:obj:`int`): The number of channels in the input.
        dim_out (:obj:`int`, *optional*): The number of channels in the output. If not given, defaults to `dim`.
        mult (:obj:`int`, *optional*, defaults to 4): The multiplier to use for the hidden dimension.
        glu (:obj:`bool`, *optional*, defaults to :obj:`False`): Whether to use GLU activation.
        dropout (:obj:`float`, *optional*, defaults to 0.0): The dropout probability to use.
    """

    def __init__(
        self,
        dim: int,
        dim_out: Optional[int] = None,
        mult: int = 4,
        glu: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = dim_out if dim_out is not None else dim

        self.proj_in = nn.Linear(dim, inner_dim * 2 if glu else inner_dim)
        self.activation = GEGLU() if glu else nn.GELU()
        self.proj_out = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out),
        )

    def forward(self, hidden_states):
        hidden_states = self.proj_in(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.proj_out(hidden_states)
        return hidden_states


class GEGLU(nn.Module):
    r"""
    A variant of the gated linear unit activation function from https://arxiv.org/abs/2002.05202.
    """

    def forward(self, hidden_states):
        hidden_states, gate = hidden_states.chunk(2, dim=-1)
        return hidden_states * F.gelu(gate)


class MultiHeadAttentionBlock(nn.Module):
    r"""
    A multi-head attention layer.

    Parameters:
        query_dim (:obj:`int`): The number of channels in the query.
        context_dim (:obj:`int`, *optional*):
            The number of channels in the context. If not given, defaults to `query_dim`.
        heads (:obj:`int`,  *optional*, defaults to 8): The number of heads to use for multi-head attention.
        dim_head (:obj:`int`,  *optional*, defaults to 64): The number of channels in each head.
        dropout (:obj:`float`, *optional*, defaults to 0.0): The dropout probability to use.
    """

    def __init__(
        self,
        query_dim: int,
        context_dim: Optional[int] = None,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
    ):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = context_dim if context_dim is not None else query_dim

        self.scale = dim_head**-0.5
        self.heads = heads
        # for slice_size > 0 the attention score computation
        # is split across the batch axis to save memory
        # You can set slice_size with `set_attention_slice`
        self._slice_size = None

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim), nn.Dropout(dropout)
        )

    def reshape_heads_to_batch_dim(self, tensor):
        batch_size, seq_len, dim = tensor.shape
        head_size = self.heads
        tensor = tensor.reshape(batch_size, seq_len, head_size, dim // head_size)
        tensor = tensor.permute(0, 2, 1, 3).reshape(
            batch_size * head_size, seq_len, dim // head_size
        )
        return tensor

    def reshape_batch_dim_to_heads(self, tensor):
        batch_size, seq_len, dim = tensor.shape
        head_size = self.heads
        tensor = tensor.reshape(batch_size // head_size, head_size, seq_len, dim)
        tensor = tensor.permute(0, 2, 1, 3).reshape(
            batch_size // head_size, seq_len, dim * head_size
        )
        return tensor

    def forward(self, hidden_states, context=None, mask=None):
        batch_size, sequence_length, _ = hidden_states.shape

        query = self.to_q(hidden_states)
        context = context if context is not None else hidden_states
        key = self.to_k(context)
        value = self.to_v(context)
        # print("context", context.shape)
        # print("query (after linear)", query.shape)
        # print("key (after linear)", key.shape)
        # print("value (after linear)", value.shape)

        dim = query.shape[-1]

        query = self.reshape_heads_to_batch_dim(query)
        key = self.reshape_heads_to_batch_dim(key)
        value = self.reshape_heads_to_batch_dim(value)

        if self._slice_size is None or query.shape[0] // self._slice_size == 1:
            hidden_states = self._attention(query, key, value, mask=mask)
        else:
            hidden_states = self._sliced_attention(
                query, key, value, sequence_length, dim, mask=mask
            )

        return self.to_out(hidden_states)

    def _attention(self, query, key, value, mask=None):
        # print("query:", query.shape)
        # print("key:", key.shape)
        # print("value:", value.shape)
        B, N, D = query.shape
        B, M, D = key.shape
        key_transpose = key.transpose(-1, -2)
        attention_scores = torch.baddbmm(
            torch.zeros(B, N, M, device=query.device),
            query,
            key_transpose,
            beta=1.0,
            alpha=self.scale,
        )
        if mask is not None:
            mask = mask.bool()
            attention_scores = attention_scores.masked_fill_(mask, float("-inf"))
        attention_probs = attention_scores.softmax(dim=-1)
        # compute attention output
        hidden_states = torch.matmul(attention_probs, value)
        # reshape hidden_states
        hidden_states = self.reshape_batch_dim_to_heads(hidden_states)
        return hidden_states

    def _sliced_attention(self, query, key, value, sequence_length, dim, mask=None):
        batch_size_attention = query.shape[0]
        hidden_states = torch.zeros(
            (batch_size_attention, sequence_length, dim // self.heads),
            device=query.device,
            dtype=query.dtype,
        )
        slice_size = (
            self._slice_size if self._slice_size is not None else hidden_states.shape[0]
        )
        for i in range(hidden_states.shape[0] // slice_size):
            start_idx = i * slice_size
            end_idx = (i + 1) * slice_size
            attn_slice = (
                torch.matmul(
                    query[start_idx:end_idx], key[start_idx:end_idx].transpose(1, 2)
                )
                * self.scale
            )  # TODO: use baddbmm for better performance
            # TODO: implement mask
            attn_slice = attn_slice.softmax(dim=-1)
            attn_slice = torch.matmul(attn_slice, value[start_idx:end_idx])

            hidden_states[start_idx:end_idx] = attn_slice

        # reshape hidden_states
        hidden_states = self.reshape_batch_dim_to_heads(hidden_states)
        return hidden_states
