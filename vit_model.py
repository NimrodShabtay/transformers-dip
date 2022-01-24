import numpy as np
import torch
import torch.nn.functional as F
# from timm.models.layers import trunc_normal_

from torch import nn
from torch import Tensor
from einops import rearrange
from einops.layers.torch import Rearrange

from utils.common_utils import get_current_iter_num

BATCH_SIZE = 1
IMG_DIM = 32
PATCH_SIZE = 2


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels: int = 3, patch_size: int = PATCH_SIZE, stride: int = 1,
                 emb_size: int = 768, img_size: int = 512):
        self.patch_size = patch_size
        self.stride = stride
        self.padding = int((patch_size - 1) / 2)
        self.dilation = 1
        patch_dim = in_channels * patch_size * patch_size
        self.L = int(np.floor(
            (img_size + 2 * self.padding - self.dilation * (self.patch_size - 1) - 1) / self.stride + 1) ** 2)
        super().__init__()
        self.projection = nn.Sequential(
            # Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
            nn.Unfold(kernel_size=self.patch_size, stride=self.stride, padding=self.padding, dilation=self.dilation),
            Rearrange('b c d -> b d c'),
            nn.Linear(patch_dim, emb_size),
            Rearrange('b d c -> b c d'),
        )
        self.positions = nn.Parameter(torch.randn(emb_size, self.L))
        # trunc_normal_(self.positions)
        # for p_ in self.projection.parameters():
        #     trunc_normal_(p_)

    def forward(self, x: Tensor) -> Tensor:
        # b, c, h, w = x.shape
        x = self.projection(x)
        x += self.positions
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size: int = 768, num_heads: int = 8, dropout: float = 0):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
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
            nn.ReLU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )


class TransformerEncoderBlock(nn.Sequential):
    def __init__(self,
                 emb_size: int = 768,
                 drop_p: float = 0.,
                 forward_expansion: int = 4,
                 forward_drop_p: float = 0.,
                 num_heads=8,
                 **kwargs):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, num_heads=num_heads),
                nn.Dropout(drop_p)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(
                    emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                nn.Dropout(drop_p)
            ))
        )


class TransformerEncoder(nn.Sequential):
    def __init__(self, depth: int = 12, **kwargs):
        super().__init__(*[TransformerEncoderBlock(**kwargs) for _ in range(depth)])


class VerboseExecution(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

        # Register a hook for each layer
        for name, layer in self.model.named_children():
            layer.__name__ = name
            layer.register_forward_hook(
                lambda layer, input, output: print(f"{layer.__name__}: {input[0].shape}->{output.shape}")
            )

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)


class PrintLayer(nn.Module):
    def __init__(self):
        super(PrintLayer, self).__init__()

    def forward(self, x):
        # Do your print / debug stuff here
        print(x.shape)
        return x


def src_mask(sz, p=0):
    device_ = 'cuda' if torch.cuda.is_available() else 'cpu'
    mask = 1 - (torch.diag(torch.from_numpy(np.random.binomial(1, 1 - p, size=sz)))).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask.to(device_)


class MaskedTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu",
                 layer_norm_eps=1e-5, batch_first=False, device=None, dtype=None):
        super(MaskedTransformerEncoderLayer, self).__init__()
        self.trans_enc_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation,
                                                          layer_norm_eps, batch_first, device, dtype)
        self.th1 = 1000
        self.th2 = 2000

        self.src_mask = None

    def forward(self, src):
        curr_iter = get_current_iter_num()
        if curr_iter < self.th1:
            p = 0
        elif self.th1 <= curr_iter < self.th2:
            p = 0.5
        elif curr_iter >= self.th2:
            p = 1
        self.src_mask = src_mask(src.shape[1], p)
        src_ = self.trans_enc_layer.forward(src, self.src_mask)
        return src_


class NormLayer(nn.Module):
    def __init__(self, output_size, kernel_size, padding, stride):
        super(NormLayer, self).__init__()
        self.fold_params = dict(kernel_size=kernel_size, dilation=1, padding=padding, stride=stride)
        self.output_size = output_size
        self.norm_mask = self.generate_norm_mask()

    def generate_norm_mask(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        norm_mask = nn.Fold(output_size=self.output_size, **self.fold_params) \
            (nn.Unfold(**self.fold_params)(torch.ones(1, 3, *self.output_size)))
        assert (norm_mask != 0).all()
        return norm_mask.to(device)

    def forward(self, src):
        norm_src = src / self.norm_mask
        return norm_src
