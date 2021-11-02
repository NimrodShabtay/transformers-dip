from models import *
from vit_model import TransformerEncoderBlock, PatchEmbedding
from models.common import *

from einops import rearrange
from einops.layers.torch import Rearrange
import torch.nn as nn
import torch


def skip_hybrid(
        num_input_channels=2, num_output_channels=3,
        num_channels_down=[16, 32, 64, 128, 128], num_channels_up=[16, 32, 64, 128, 128],
        num_channels_skip=[4, 4, 4, 4, 4],
        filter_size_down=3, filter_size_up=3, filter_skip_size=1,
        need_sigmoid=True, need_bias=True,
        pad='zero', upsample_mode='nearest', downsample_mode='stride', act_fun='LeakyReLU',
        need1x1_up=True, img_sz=512):
    """Assembles encoder-decoder with skip connections.

    Arguments:
        act_fun: Either string 'LeakyReLU|Swish|ELU|none' or module (e.g. nn.ReLU)
        pad (string): zero|reflection (default: 'zero')
        upsample_mode (string): 'nearest|bilinear' (default: 'nearest')
        downsample_mode (string): 'stride|avg|max|lanczos2' (default: 'stride')

    """
    assert len(num_channels_down) == len(num_channels_up) == len(num_channels_skip)

    n_scales = len(num_channels_down)

    if not (isinstance(upsample_mode, list) or isinstance(upsample_mode, tuple)):
        upsample_mode = [upsample_mode] * n_scales

    if not (isinstance(downsample_mode, list) or isinstance(downsample_mode, tuple)):
        downsample_mode = [downsample_mode] * n_scales

    if not (isinstance(filter_size_down, list) or isinstance(filter_size_down, tuple)):
        filter_size_down = [filter_size_down] * n_scales

    if not (isinstance(filter_size_up, list) or isinstance(filter_size_up, tuple)):
        filter_size_up = [filter_size_up] * n_scales

    last_scale = n_scales - 1
    transformer_start_level = 0
    use_transformer_skip = True
    cur_depth = None

    model = nn.Sequential()
    model_tmp = model

    input_depth = num_input_channels

    for i in range(len(num_channels_down)):
        last_spatial_dim = img_sz // 2 ** i
        deeper = nn.Sequential()
        skip = nn.Sequential()

        if num_channels_skip[i] != 0:
            model_tmp.add(Concat(1, skip, deeper))
        else:
            model_tmp.add(deeper)

        model_tmp.add(bn(num_channels_skip[i] + (num_channels_up[i + 1] if i < last_scale else num_channels_down[i])))

        if num_channels_skip[i] != 0:
            if use_transformer_skip:
                num_heads = num_channels_skip[i] if num_channels_skip[i] < 8 else 8
                t_block = transformer_block(input_depth, num_channels_skip[i], 1, num_heads)
                skip.add(t_block)
                skip.add(Rearrange('b (h w) (c)-> b c (h) (w)', h=last_spatial_dim, w=last_spatial_dim))
            else:
                skip.add(conv(input_depth, num_channels_skip[i], filter_skip_size, bias=need_bias, pad=pad))

            skip.add(bn(num_channels_skip[i]))
            skip.add(act(act_fun))

        # skip.add(Concat(2, GenNoise(nums_noise[i]), skip_part))
        if i >= transformer_start_level:
            deeper.add(transformer_block(input_depth, num_channels_down[i], patch_size=2))
            deeper.add(Rearrange('b (h w) (c)-> b c (h) (w)', h=last_spatial_dim//2, w=last_spatial_dim//2))
        else:
            deeper.add(conv(input_depth, num_channels_down[i], filter_size_down[i], 2, bias=need_bias, pad=pad,
                            downsample_mode=downsample_mode[i]))

        deeper.add(bn(num_channels_down[i]))
        deeper.add(act(act_fun))

        if i >= transformer_start_level:
            deeper.add(transformer_block(num_channels_down[i], num_channels_down[i]))
            deeper.add(Rearrange('b (h w) (c)-> b c (h) (w)', h=last_spatial_dim//2, w=last_spatial_dim//2))
        else:
            deeper.add(conv(num_channels_down[i], num_channels_down[i], filter_size_down[i], bias=need_bias, pad=pad))

        deeper.add(bn(num_channels_down[i]))
        deeper.add(act(act_fun))

        deeper_main = nn.Sequential()

        if i == len(num_channels_down) - 1:
            # The deepest
            k = num_channels_down[i]
        else:
            deeper.add(deeper_main)
            k = num_channels_up[i + 1]

        deeper.add(nn.Upsample(scale_factor=2, mode=upsample_mode[i]))
        if i == len(num_channels_down) - 1:
            model_tmp.add(transformer_block(num_channels_skip[i] + k, num_channels_up[i]))
            model_tmp.add(Rearrange('b (h w) (c)-> b c (h) (w)', h=last_spatial_dim, w=last_spatial_dim))

        model_tmp.add(bn(num_channels_up[i]))
        model_tmp.add(act(act_fun))

        if need1x1_up:
            # model_tmp.add(conv(num_channels_up[i], num_channels_up[i], 1, bias=need_bias, pad=pad))
            model_tmp.add(transformer_block(num_channels_up[i], num_channels_up[i]))
            model_tmp.add(Rearrange('b (h w) (c)-> b c (h) (w)', h=last_spatial_dim//2, w=last_spatial_dim//2))
            model_tmp.add(bn(num_channels_up[i]))
            model_tmp.add(act(act_fun))

        input_depth = num_channels_down[i]
        model_tmp = deeper_main

    # model.add(conv(num_channels_up[0], num_output_channels, 1, bias=need_bias, pad=pad))
    model.add(transformer_block(num_channels_up[0], num_output_channels, 1, num_heads=num_output_channels))
    model.add(Rearrange('b (h w) (c)-> b c (h) (w)', h=img_sz, w=img_sz))
    if need_sigmoid:
        model.add(nn.Sigmoid())

    return model


def transformer_block(input_channels, embedding_size, patch_size=1, num_heads=8):
    t_block = nn.Sequential()
    t_block.add(PatchEmbedding(in_channels=input_channels, patch_size=patch_size, emb_size=embedding_size))
    t_block.add(TransformerEncoderBlock(emb_size=embedding_size, num_heads=num_heads))
    return t_block
