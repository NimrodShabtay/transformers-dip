import sys
import torch
import torch.nn as nn

from utils.common_utils import crop_image, get_image, pil_to_np, np_to_torch
from utils.denoising_utils import get_noisy_image

sys.path.append('../SwinIR/models/')
from network_swinir import SwinIR

device = 'cuda' if torch.cuda.is_available() else 'cpu'
sigma = 25
sigma_ = sigma / 255.
fname = ['data/denoising/F16_GT.png', 'data/inpainting/kate.png'][0]
img_pil = crop_image(get_image(fname, (128, 128))[0], d=32)
img_np = pil_to_np(img_pil)

img_noisy_pil, img_noisy_np = get_noisy_image(img_np, sigma_)

height, width = img_noisy_pil.size
window_size = 4

m = SwinIR(upscale=1, img_size=(height, width),
           window_size=window_size, img_range=1., depths=[6, 6],
           embed_dim=60, num_heads=[6, 6], mlp_ratio=2, upsampler=None)
m.to(device)

result = m(np_to_torch(img_noisy_np).to(device))
pass


class SwinIR_dip(SwinIR):
    def __init__(self, img_size=64, patch_size=1, in_chans=3,
                 embed_dim=96, depths=[6, 6, 6, 6], num_heads=[6, 6, 6, 6],
                 window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, upscale=1, img_range=1., upsampler='', resi_connection='1conv',
                 **kwargs):
        super(SwinIR_dip, self).__init__(img_size, patch_size, in_chans,
                 embed_dim, depths, num_heads,
                 window_size, mlp_ratio, qkv_bias, qk_scale,
                 drop_rate, attn_drop_rate, drop_path_rate,
                 norm_layer, ape, patch_norm,
                 use_checkpoint, upscale, img_range, upsampler, resi_connection,
                 **kwargs)
        self.linear_last = nn.Linear(self.embed_dim, in_chans)
        self.sigmoid = nn.Sigmoid()

    def forward_transformer(self, x):
        x_size = (x.shape[2], x.shape[3])
        x_begin = self.patch_embed(x)
        if self.ape:
            x_begin = x_begin + self.absolute_pos_embed
        x = self.pos_drop(x_begin)

        for layer in self.layers:
            x = layer(x, x_size)

        x = self.norm(x)  # B L C
        x = x + x_begin
        x = self.linear_last(x)

        return x

    def forward(self, x):
        x = self.check_image_size(x)

        x_first = self.conv_first(x)
        res = self.forward_features(x_first)

        self.sigmoid(res)
        return res

