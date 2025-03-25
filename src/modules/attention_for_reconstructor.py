# This script is provided by the FYP24 project group.
# This script is adapted from attention.py, which is provided by authors of FontDiffuser.

from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F


### Spatial Transformer ###

class SpatialTransformer(nn.Module):
    """
    Transformer block for image-like data. First, project the input (aka embedding) and reshape to b, t, d. Then apply
    standard transformer action. Finally, reshape to image.

    Parameters:
        in_channels (:obj:`int`): The number of channels in the input and output.
        n_heads (:obj:`int`): The number of heads to use for multi-head attention.
        d_head (:obj:`int`): The number of channels in each head.
        depth (:obj:`int`, *optional*, defaults to 1): The number of layers of Transformer blocks to use.
        dropout (:obj:`float`, *optional*, defaults to 0.1): The dropout probability to use.
        context_dim (:obj:`int`, *optional*): The number of context dimensions to use.
    """

    def __init__(
        self,
        in_channels: int,
        n_heads: int,
        d_head: int,
        query_dim: int,
        depth: int = 1,
        dropout: float = 0.0,
        num_groups: int = 32,
        context_channels: Optional[int] = None,
        context_dim: Optional[int] = None,
        valid_ratio: tuple[int, int]=(1, 1),
    ):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_head
        self.in_channels = in_channels
        d_embed = n_heads * d_head
        self.norm = torch.nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)

        # self.proj_in = nn.Conv2d(in_channels, d_embed, kernel_size=1, stride=1, padding=0)

        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    query_channels=in_channels,
                    query_dim=query_dim,
                    n_heads=n_heads,
                    d_head=d_head,
                    dropout=dropout,
                    context_channels=context_channels,
                    context_dim=context_dim,
                    valid_ratio=valid_ratio,
                )
                for d in range(depth)
            ]
        )

        # self.proj_out = nn.Conv2d(d_embed, in_channels, kernel_size=1, stride=1, padding=0)

    def _set_attention_slice(self, slice_size):
        for block in self.transformer_blocks:
            assert block is torch.Module
            block._set_attention_slice(slice_size)

    def forward(self, hidden_states, context=None):
        # note: if no context is given, cross-attention defaults to self-attention
        residual = hidden_states
        hidden_states = self.norm(hidden_states)
        # hidden_states = self.proj_in(hidden_states)
        # print("hidden_states", hidden_states.shape)
        # batch, channel, height, weight = hidden_states.shape
        # inner_dim = channel
        # hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch, height * weight, inner_dim)  # here change the shape torch.Size([1, 4096, 128])
        # hidden_states = hidden_states.reshape(batch, channel, height * weight)
        for block in self.transformer_blocks:
            hidden_states = block(hidden_states, context=context)
        # hidden_states = hidden_states.reshape(batch, height, weight, inner_dim).permute(0, 3, 1, 2)
        # hidden_states = hidden_states.reshape(batch, channel, height, weight)
        # hidden_states = self.proj_out(hidden_states)
        return hidden_states + residual


class BasicTransformerBlock(nn.Module):
    r"""
    A basic Transformer block.

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
        query_channels: int,
        query_dim: int,
        n_heads: int,
        d_head: int,
        dropout=0.0,
        context_channels: Optional[int] = None,
        context_dim: Optional[int] = None,
        valid_ratio: tuple[int, int]=(1, 1),
        gated_ff: bool = True,
        checkpoint: bool = True,
    ):
        super().__init__()
        self.attn1 = CrossAttention(
            query_dim=query_dim, heads=n_heads, dim_head=d_head, dropout=dropout
        )  # is a self-attention
        self.ff = FeedForward(query_dim, dropout=dropout, glu=gated_ff, mult=2)
        self.attn2 = CrossAttention(
            query_dim=query_dim, context_dim=context_dim, heads=n_heads, dim_head=d_head, dropout=dropout
        )  # is self-attn if context is none
        self.norm1 = nn.LayerNorm(query_dim)
        self.norm2 = nn.LayerNorm(query_dim)
        self.norm3 = nn.LayerNorm(query_dim)
        self.checkpoint = checkpoint

        # Prepare attention mask 1
        valid_query_channels = query_channels * valid_ratio[0] // valid_ratio[1]
        attn_mask_1 = torch.ones(query_channels, query_channels, dtype=torch.int, requires_grad=False)
        attn_mask_1[:valid_query_channels, :valid_query_channels] = 0
        self.register_buffer("attn_mask_1", attn_mask_1)

        # Prepare attention mask 2
        attn_mask_2 = attn_mask_1
        if context_channels is not None:
            valid_context_channels = context_channels * valid_ratio[0] // valid_ratio[1]
            attn_mask_2 = torch.ones(query_channels, context_channels, dtype=torch.int, requires_grad=False)
            attn_mask_2[:valid_query_channels, :valid_context_channels] = 0
        self.register_buffer("attn_mask_2", attn_mask_2)

    def _set_attention_slice(self, slice_size):
        self.attn1._slice_size = slice_size
        self.attn2._slice_size = slice_size

    def forward(self, hidden_states, context=None):
        hidden_states = hidden_states.contiguous() if hidden_states.device.type == "mps" else hidden_states
        # print("hidden_states (before attn1)", hidden_states.shape)
        hidden_states = self.attn1(self.norm1(hidden_states), mask=self.attn_mask_1) + hidden_states
        # print("hidden_states (after attn1)", hidden_states.shape)
        hidden_states = self.attn2(self.norm2(hidden_states), context=context, mask=self.attn_mask_2) + hidden_states
        # print("hidden_states (after attn2)", hidden_states.shape)
        hidden_states = self.ff(self.norm3(hidden_states)) + hidden_states
        # print("hidden_states (after ff)", hidden_states.shape)
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
        self, dim: int, dim_out: Optional[int] = None, mult: int = 4, glu: bool = False, dropout: float = 0.0
    ):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = dim_out if dim_out is not None else dim
        project_in = GEGLU(dim, inner_dim)

        self.net = nn.Sequential(project_in, nn.Dropout(dropout), nn.Linear(inner_dim, dim_out))

    def forward(self, hidden_states):
        return self.net(hidden_states)


class GEGLU(nn.Module):
    r"""
    A variant of the gated linear unit activation function from https://arxiv.org/abs/2002.05202.

    Parameters:
        dim_in (:obj:`int`): The number of channels in the input.
        dim_out (:obj:`int`): The number of channels in the output.
    """

    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, hidden_states):
        hidden_states, gate = self.proj(hidden_states).chunk(2, dim=-1)
        return hidden_states * F.gelu(gate)


class CrossAttention(nn.Module):
    r"""
    A cross attention layer.

    Parameters:
        query_dim (:obj:`int`): The number of channels in the query.
        context_dim (:obj:`int`, *optional*):
            The number of channels in the context. If not given, defaults to `query_dim`.
        heads (:obj:`int`,  *optional*, defaults to 8): The number of heads to use for multi-head attention.
        dim_head (:obj:`int`,  *optional*, defaults to 64): The number of channels in each head.
        dropout (:obj:`float`, *optional*, defaults to 0.0): The dropout probability to use.
    """

    def __init__(
        self, query_dim: int, context_dim: Optional[int] = None, heads: int = 8, dim_head: int = 64, dropout: float = 0.0
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

        self.to_out = nn.Sequential(nn.Linear(inner_dim, query_dim), nn.Dropout(dropout))

    def reshape_heads_to_batch_dim(self, tensor):
        batch_size, seq_len, dim = tensor.shape
        head_size = self.heads
        tensor = tensor.reshape(batch_size, seq_len, head_size, dim // head_size)
        tensor = tensor.permute(0, 2, 1, 3).reshape(batch_size * head_size, seq_len, dim // head_size)
        return tensor

    def reshape_batch_dim_to_heads(self, tensor):
        batch_size, seq_len, dim = tensor.shape
        head_size = self.heads
        tensor = tensor.reshape(batch_size // head_size, head_size, seq_len, dim)
        tensor = tensor.permute(0, 2, 1, 3).reshape(batch_size // head_size, seq_len, dim * head_size)
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
            hidden_states = self._sliced_attention(query, key, value, sequence_length, dim, mask=mask)

        return self.to_out(hidden_states)

    def _attention(self, query, key, value, mask=None):
        # print("query:", query.shape)
        # print("key:", key.shape)
        # print("value:", value.shape)
        B, N, D = query.shape
        B, M, D = key.shape
        key_transpose = key.transpose(-1, -2)
        attention_scores = torch.baddbmm(torch.zeros(B, N, M, device=query.device), query, key_transpose, beta=1.0, alpha=self.scale)
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
            (batch_size_attention, sequence_length, dim // self.heads), device=query.device, dtype=query.dtype
        )
        slice_size = self._slice_size if self._slice_size is not None else hidden_states.shape[0]
        for i in range(hidden_states.shape[0] // slice_size):
            start_idx = i * slice_size
            end_idx = (i + 1) * slice_size
            attn_slice = (
                torch.matmul(query[start_idx:end_idx], key[start_idx:end_idx].transpose(1, 2)) * self.scale
            )  # TODO: use baddbmm for better performance
            # TODO: implement mask
            attn_slice = attn_slice.softmax(dim=-1)
            attn_slice = torch.matmul(attn_slice, value[start_idx:end_idx])

            hidden_states[start_idx:end_idx] = attn_slice

        # reshape hidden_states
        hidden_states = self.reshape_batch_dim_to_heads(hidden_states)
        return hidden_states

### Channel Attention Block ###

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            # nn.ReLU(inplace=True),
            nn.SiLU(),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class Mish(torch.nn.Module):
    def forward(self, hidden_states):
        return hidden_states * torch.tanh(torch.nn.functional.softplus(hidden_states))


class ChannelAttnBlock(nn.Module):
    """This is the Channel Attention in MCA.
    """
    def __init__(
        self,
        in_channels,
        out_channels,
        groups=32,
        groups_out=None,
        eps=1e-6,
        non_linearity="swish",
        channel_attn=False,
        reduction=32):
        super().__init__()

        if groups_out is None:
            groups_out = groups

        self.norm1 = nn.GroupNorm(num_groups=groups, num_channels=in_channels, eps=eps, affine=True)
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1)

        if non_linearity == "swish":
            self.nonlinearity = lambda x: F.silu(x)
        elif non_linearity == "mish":
            self.nonlinearity = Mish()
        elif non_linearity == "silu":
            self.nonlinearity = nn.SiLU()
        
        self.channel_attn = channel_attn
        if self.channel_attn:
            # SE Attention
            self.se_channel_attn = SELayer(channel=in_channels, reduction=reduction)

        # Down channel: Use the conv1*1 to down the channel wise
        self.norm3 = nn.GroupNorm(num_groups=groups, num_channels=in_channels, eps=eps, affine=True)
        self.down_channel = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1) # conv1*1

    def forward(self, inputs: torch.Tensor):

        # concat_feature = torch.cat(inputs, dim=1)
        # hidden_states = concat_feature

        concat_feature = inputs
        hidden_states = concat_feature

        # print("hidden_states (before norm1)", hidden_states.shape)
        hidden_states = self.norm1(hidden_states)
        # print("hidden_states (after norm1)", hidden_states.shape)
        hidden_states = self.nonlinearity(hidden_states)
        # print("hidden_states (after nonlinearity)", hidden_states.shape)
        hidden_states = self.conv1(hidden_states)
        # print("hidden_states (after conv1)", hidden_states.shape)

        if self.channel_attn:
            hidden_states = self.se_channel_attn(hidden_states)
            hidden_states = hidden_states + concat_feature

        # Down channel
        # print("hidden_states (before down_channel)", hidden_states.shape)
        hidden_states = self.norm3(hidden_states)
        # print("hidden_states (after norm3)", hidden_states.shape)
        hidden_states = self.nonlinearity(hidden_states)
        # print("hidden_states (after nonlinearity)", hidden_states.shape)
        hidden_states = self.down_channel(hidden_states)
        # print("hidden_states (after down_channel)", hidden_states.shape)

        return hidden_states
