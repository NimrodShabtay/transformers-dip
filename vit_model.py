import torch
import torch.nn.functional as F
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

from torch import nn
from torch import Tensor
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce


BATCH_SIZE = 1
IMG_DIM = 32
PATCH_SIZE = 2


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels: int = 3, patch_size: int = PATCH_SIZE, emb_size: int = 768, img_size=IMG_DIM):
        self.patch_size = patch_size
        patch_dim = in_channels * patch_size * patch_size
        super().__init__()
        self.projection = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
            nn.Linear(patch_dim, emb_size)
        )

    def forward(self, x: Tensor) -> Tensor:
        b, c, h, w = x.shape
        x = self.projection(x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, input_size, emb_size: int = 768, num_heads: int = 8, dropout: float = 0):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(input_size, emb_size)
        self.queries = nn.Linear(input_size, emb_size)
        self.values = nn.Linear(input_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        # split keys, queries and values in num_heads
        queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)
        keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.num_heads)
        values = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.num_heads)
        # sum up over the last axis
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)  # batch, num_heads, query_len, key_len
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)

        scaling = self.emb_size ** (1 / 2)
        att = F.softmax(energy, dim=-1) / scaling
        att = self.att_drop(att)
        # sum up over the third axis
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out


class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x


class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size: int, expansion: int = 4, drop_p: float = 0.):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )


class TransformerEncoderBlock(nn.Sequential):
    def __init__(self,
                 input_size,
                 emb_size: int = 768,
                 drop_p: float = 0.,
                 forward_expansion: int = 4,
                 forward_drop_p: float = 0.,
                 num_heads=8,
                 **kwargs):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(input_size, emb_size, num_heads=num_heads),
                nn.Dropout(drop_p)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(
                    emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                nn.Dropout(drop_p)
            )
            ))


class TransformerEncoder(nn.Sequential):
    def __init__(self, depth: int = 12, **kwargs):
        super().__init__(*[TransformerEncoderBlock(**kwargs) for _ in range(depth)])


# class ClassificationHead(nn.Sequential):
#     def __init__(self, emb_size: int = 768, n_classes: int = 1000):
#         super().__init__(
#             Reduce('b n e -> b e', reduction='mean'),
#             nn.LayerNorm(emb_size),
#             nn.Linear(emb_size, n_classes))


# class ViT(nn.Sequential):
#     def __init__(self,
#                  in_channels: int = 3,
#                  patch_size: int = PATCH_SIZE,
#                  emb_size: int = 768,
#                  img_size: int = IMG_DIM,
#                  depth: int = 6,
#                  n_classes: int = 10,
#                  **kwargs):
#         super().__init__(
#             PatchEmbedding(in_channels, patch_size, emb_size, img_size),
#             TransformerEncoder(depth, emb_size=emb_size, **kwargs),
#             ClassificationHead(emb_size, n_classes)
#         )
