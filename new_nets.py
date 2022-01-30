from vit_model import PatchEmbedding, PrintLayer, MaskedTransformerEncoderLayer, PatchUnEmbedding
from models.common import *

from einops.layers.torch import Rearrange
import torch.nn as nn
import logging
from collections import OrderedDict

# TODO: Refactor function signature + docstring

norm1d = [nn.InstanceNorm1d, nn.BatchNorm1d][1]
logger = logging.getLogger('exp_logger')


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

    if not (isinstance(downsample_mode, list) or isinstance(downsample_mode, tuple)):
        downsample_mode = [downsample_mode] * n_scales

    if not (isinstance(filter_size_down, list) or isinstance(filter_size_down, tuple)):
        filter_size_down = [filter_size_down] * n_scales

    if not (isinstance(filter_size_up, list) or isinstance(filter_size_up, tuple)):
        filter_size_up = [filter_size_up] * n_scales

    last_scale = n_scales - 1
    num_heads = 8
    transformer_blocks_start = 2
    transformer_activation = 'relu'
    patch_sz = 2
    dropout_rate = 0.0
    stride = patch_sz if transformer_blocks_start != -1 else patch_sz // 2
    assert transformer_blocks_start <= n_scales, "conv_block_ends index must be smaller than n_scales, or -1 for non-conv blocks"

    logger.info(
        'Num heads: {} conv_block_ends: {} norm: {}\n transformer act: {} patch size: {} dropout rate: {} patch stride: {}'.format(
            num_heads, transformer_blocks_start, norm1d.__name__, transformer_activation, patch_sz, dropout_rate, stride))

    logger.info('Mask self attention')
    model = nn.Sequential()
    model_tmp = model

    input_depth = num_input_channels
    org_spatial_dim = img_sz
    if transformer_blocks_start < 0:
        patch_emb_layer = PatchEmbedding(input_depth, patch_sz, stride, num_channels_down[0], img_sz, do_project=True)
        model_tmp.add(patch_emb_layer)
        input_depth = num_channels_down[0]
        org_spatial_dim = int(np.sqrt(patch_emb_layer.L))

    current_spatial_dim = org_spatial_dim
    j = 0
    for i in range(len(num_channels_down)):
        deeper = nn.Sequential()
        skip = nn.Sequential()
        if num_channels_skip[i] != 0:
            if i < transformer_blocks_start:
                model_tmp.add(Concat(1, skip, deeper))
            else:
                if i == transformer_blocks_start:
                    # Finish with conv blocks, project to 1D for transformer blocks
                    patch_emb_layer = PatchEmbedding(in_channels=num_channels_down[i], patch_size=patch_sz,
                                                     stride=stride, emb_size=num_channels_down[i],
                                                     img_size=current_spatial_dim // 2, do_project=False)
                    org_spatial_dim = int(np.sqrt(patch_emb_layer.L))
                    # Adapt sizes after tokenize the feature maps
                    current_spatial_dim = org_spatial_dim
                    j = 1
                    num_channels_down[i:] = [patch_sz * patch_sz * num_channels_down[k] for k in
                                             range(transformer_blocks_start, len(num_channels_down))]
                    num_channels_up[i:] = [patch_sz * patch_sz * num_channels_up[k] for k in
                                           range(transformer_blocks_start, len(num_channels_up))]
                    model_tmp.add(patch_emb_layer)
                model_tmp.add(Concat1d(1, skip, deeper))
        else:
            model_tmp.add(deeper)

        channels_ = num_channels_skip[i] + (num_channels_up[i + 1] if i < last_scale else num_channels_down[i])
        if i < transformer_blocks_start:
            model_tmp.add(bn(channels_))
        else:
            model_tmp.add(norm1d(channels_))

        if num_channels_skip[i] != 0:
            if i < transformer_blocks_start:
                skip.add(conv(input_depth, num_channels_skip[i], filter_skip_size, bias=need_bias, pad=pad))
                skip.add(bn(num_channels_skip[i]))
            else:
                skip.add(Rearrange('b c l -> b l c'))
                skip.add(nn.Linear(num_channels_down[i], num_channels_skip[i]))
                skip.add(Rearrange('b l c -> b c l'))
                skip.add(norm1d(num_channels_skip[i]))

            skip.add(act(act_fun))

        if i < transformer_blocks_start:
            deeper.add(conv(input_depth, num_channels_down[i], filter_size_down[i], 2, bias=need_bias, pad=pad,
                            downsample_mode=downsample_mode[i]))
            deeper.add(bn(num_channels_down[i]))
            deeper.add(act(act_fun))

            deeper.add(conv(num_channels_down[i], num_channels_down[i], filter_size_down[i], bias=need_bias, pad=pad))
            deeper.add(bn(num_channels_down[i]))
            deeper.add(act(act_fun))
        else:
            # If the the first downsampling is done in the patch embedding layer
            # skip it in the last level for both upsampling and downsampling
            if 0 < i < n_scales:
                deeper.add(downsampling_block(dim=current_spatial_dim, scale_factor=2))

            deeper.add(
                transformer_block(num_channels_down[i],
                                  num_channels_down[i],
                                  num_heads, transformer_activation, dropout_rate))
            deeper.add(norm1d(num_channels_down[i]))
            deeper.add(act(act_fun))

            # deeper.add(
            #     transformer_block(num_channels_down[i],
            #                       num_channels_down[i],
            #                       num_heads, transformer_activation, dropout_rate))
            # deeper.add(norm1d(num_channels_down[i]))
            # deeper.add(act(act_fun))

        deeper_main = nn.Sequential()

        if i == len(num_channels_down) - 1:
            # The deepest
            k = num_channels_down[i]
        else:
            deeper.add(deeper_main)
            k = num_channels_up[i + 1]

        current_channels_count = num_channels_skip[i] + k
        if i < transformer_blocks_start:
            deeper.add(nn.Upsample(scale_factor=2, mode='bilinear'))
            model_tmp.add(PrintLayer())
            model_tmp.add(
                conv(current_channels_count, num_channels_up[i], filter_size_up[i], 1, bias=need_bias, pad=pad))
            model_tmp.add(bn(num_channels_up[i]))
        else:  # Transformer part
            # If the the first downsampling is done in the patch embedding layer
            # skip it in the last level for both upsampling and downsampling
            if i > 0:
                deeper.add(upsampling_block(dim=current_spatial_dim // 2, scale_factor=2))

            model_tmp.add(
                transformer_block(current_channels_count,
                                  num_channels_up[i],
                                  num_heads, transformer_activation, dropout_rate))
            model_tmp.add(norm1d(num_channels_up[i]))

        model_tmp.add(act(act_fun))

        if need1x1_up:
            if i < transformer_blocks_start:
                model_tmp.add(conv(num_channels_up[i], num_channels_up[i], 1, bias=need_bias, pad=pad))
                model_tmp.add(bn(num_channels_up[i]))
                model_tmp.add(act(act_fun))

            else:
                # model_tmp.add(
                #     transformer_block(num_channels_up[i],
                #                       num_channels_up[i],
                #                       num_heads, transformer_activation, dropout_rate))
                # model_tmp.add(norm1d(num_channels_up[i]))
                # model_tmp.add(act(act_fun))
                pass

        if i == transformer_blocks_start:
            model_tmp.add(PatchUnEmbedding(emb_size=num_channels_up[i-1],
                                        patch_size=patch_sz, stride=stride,
                                        out_size=current_spatial_dim * patch_sz))

        input_depth = num_channels_down[i]
        model_tmp = deeper_main
        current_spatial_dim = org_spatial_dim // 2 ** j
        j += 1

    if transformer_blocks_start > 0:
        model.add(conv(num_channels_up[0], num_output_channels, 1, bias=need_bias, pad=pad))
    else:
        model.add(Rearrange('b c l -> b l c'))
        model.add(nn.Linear(num_channels_down[0], patch_sz * patch_sz * num_output_channels))
        model.add(Rearrange('b l c -> b c l'))
        model.add(PatchUnEmbedding(emb_size=num_output_channels, patch_size=patch_sz, stride=stride, out_size=img_sz))

    if need_sigmoid:
        model.add(nn.Sigmoid())

    return model


def transformer_block(input_dims, output_dims, num_heads, transformer_act, dropout_rate, ff_expansion=4):
    d = OrderedDict(
        [
            ('transformer_rearrange_before', Rearrange('b c l -> b l c')),
            # ('transformer_msa', nn.TransformerEncoderLayer(input_dims, num_heads, ff_expansion * input_dims,
            #                                                dropout_rate, activation=transformer_act, batch_first=True)),
            # ('transformer_msa', TransformerEncoderBlock(input_dims)),
            ('transformer_msa', MaskedTransformerEncoderLayer(input_dims, num_heads, ff_expansion * input_dims,
                                                              dropout_rate, activation=transformer_act,
                                                              batch_first=True)),
        ]
    )
    if input_dims != output_dims:
        d.update({'transformer_fix_dim': nn.Linear(input_dims, output_dims)})

    d.update({'transformer_rearrange_after': Rearrange('b l c -> b c l')})

    t_block = nn.Sequential(d)
    return t_block


def upsampling_block(dim, scale_factor):
    block = nn.Sequential()
    block.add(Rearrange('b c (w h) -> b c (w) (h)', w=dim, h=dim))
    block.add(nn.Upsample(scale_factor=scale_factor, mode='bilinear'))
    block.add(Rearrange('b c (w) (h) -> b c (w h)', w=dim * scale_factor, h=dim * scale_factor))
    return block


def downsampling_block(dim, scale_factor):
    block = nn.Sequential()
    block.add(PrintLayer())
    block.add(Rearrange('b c (w h) -> b c (w) (h)', w=dim, h=dim))
    block.add(PrintLayer())
    block.add(nn.MaxPool2d(scale_factor))
    block.add(Rearrange('b c (w) (h) -> b c (w h)', w=dim // scale_factor, h=dim // scale_factor))
    return block

