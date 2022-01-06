from __future__ import print_function
# from torchinfo import summary

from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from datetime import datetime
import logging
import sys
from utils.denoising_utils import *
from models import *


torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

imsize = -1
PLOT = True
sigma = 25
sigma_ = sigma / 255.

now = datetime.now()

params_dict = {
    'org': {
        'model': 'skip',
        'filters': 128,
        'scales': 5,
        'title': 'Original',
        'filename': 'original'
    },
    'transformer': {
        'model': 'skip_hybrid',
        'filters': 16,
        'scales': 5,
        'title': 'Transformer ',
        'filename': 'transformer',
        'save_dir': './exps/{}_{}_{}_{}_{}'.format(now.year, now.month, now.day, now.hour, now.minute)
    }
}

EXP = 'transformer'
d = params_dict[EXP]
os.mkdir(d['save_dir'])
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(d['save_dir'], 'log.txt')),
        logging.StreamHandler(sys.stdout)
    ]
)
if __name__ == '__main__':
    logger = logging.getLogger('exp_logger')
    fname = ['data/denoising/F16_GT.png', 'data/inpainting/kate.png'][0]
    if fname == 'data/denoising/snail.jpg':
        img_noisy_pil = crop_image(get_image(fname, imsize)[0], d=8)
        img_noisy_np = pil_to_np(img_noisy_pil)

        # As we don't have ground truth
        img_pil = img_noisy_pil
        img_np = img_noisy_np

        if PLOT:
            plot_image_grid([img_np], 4, 5)

    elif fname in ['data/denoising/F16_GT.png', 'data/inpainting/kate.png']:
        # Add synthetic noise
        img_pil = crop_image(get_image(fname, imsize)[0], d=32)
        # img_pil = img_pil.resize((128, 128), resample=Image.BICUBIC)
        img_np = pil_to_np(img_pil)

        img_noisy_pil, img_noisy_np = get_noisy_image(img_np, sigma_)

        # if PLOT:
        #     plot_image_grid([img_np, img_noisy_np], factor=4, nrow=1, count='org')

    else:
        assert False

    INPUT = 'noise'  # 'meshgrid'
    pad = 'reflection'
    OPT_OVER = 'net'  # 'net,input'

    reg_noise_std = 1. / 30.  # set to 1./20. for sigma=50
    LR = 0.01
    WD = 0.3  # like in ViT, default for Pytorch 0.01

    OPTIMIZER = 'adamW'  # 'LBFGS'
    show_every = 100
    exp_weight = 0.99
    logger.info('Optimizer: {} LR: {} WD: {}'.format(OPTIMIZER, LR, WD))

    if fname == 'data/denoising/snail.jpg':
        num_iter = 2400
        input_depth = 3
        figsize = 5

        net = skip(
            input_depth, 3,
            num_channels_down=[8, 16, 32, 64, 128],
            num_channels_up=[8, 16, 32, 64, 128],
            num_channels_skip=[0, 0, 0, 4, 4],
            upsample_mode='bilinear',
            need_sigmoid=True, need_bias=True, pad=pad, act_fun='LeakyReLU')

        net = net.type(dtype)

    elif fname in ['data/denoising/F16_GT.png', 'data/inpainting/kate.png']:
        num_iter = 1000
        input_depth = 32
        figsize = 4
        net = get_net(input_depth, d['model'],
                      pad, upsample_mode='linear',
                      skip_n33d=d['filters'], skip_n33u=d['filters'], skip_n11=8,
                      num_scales=d['scales'], img_sz=img_pil.size[0]).type(dtype)

        logger.info('Num scales: {} Num channels in each level: {}'.format(d['scales'], d['filters']))

        # net_ref = get_net(input_depth, 'skip', pad,
        #               skip_n33d=64,
        #               skip_n33u=64,
        #               skip_n11=4,
        #               num_scales=4,
        #               upsample_mode='bilinear').type(dtype)
        # print(net)
        # torch.save(net, 'model.pth')
        # summary(net, (1, input_depth, img_pil.size[0], img_pil.size[1]))

    net_input = get_noise(input_depth, INPUT, (img_pil.size[1], img_pil.size[0])).type(dtype).detach()
    # Compute number of parameters
    s = sum([np.prod(list(p.size())) for p in net.parameters()])
    logger.info('Number of params: %d' % s)

    # Loss
    mse = torch.nn.MSELoss().type(dtype)
    img_noisy_torch = np_to_torch(img_noisy_np).type(dtype)

    net_input_saved = net_input.detach().clone()
    noise = net_input.detach().clone()
    out_avg = None
    last_net = None
    psnr_noisy_last = 0
    psnr_gt_last = 0
    i = 0
    psnr_gt_vals = []
    mse_vals = []
    psnr_noisy_gt_vals = []

    def closure():
        global i, out_avg, psnr_noisy_last, last_net, net_input, psnr_gt_vals, mse_vals, psnr_gt_last

        if reg_noise_std > 0:
            net_input = net_input_saved + (noise.normal_() * reg_noise_std)
        out = net(net_input)
        # make_dot(out.mean(), params=dict(net.named_parameters())).render("attached", format='png')

        # Smoothing
        if out_avg is None:
            out_avg = out.detach()
        else:
            out_avg = out_avg * exp_weight + out.detach() * (1 - exp_weight)

        total_loss = mse(out, img_noisy_torch)
        mse_vals.append(total_loss.item())
        total_loss.backward()
        # plot_grad_flow(net.named_parameters())

        psnr_noisy = compare_psnr(img_noisy_np, out.detach().cpu().numpy()[0])
        psnr_gt = compare_psnr(img_np, out.detach().cpu().numpy()[0])
        psnr_gt_sm = compare_psnr(img_np, out_avg.detach().cpu().numpy()[0])

        psnr_gt_vals.append(psnr_gt)
        psnr_noisy_gt_vals.append(psnr_noisy)

        # Note that we do not have GT for the "snail" example
        # So 'PSRN_gt', 'PSNR_gt_sm' make no sense
        if PLOT and (i % show_every == 0):
            logger.info('Iteration %05d    Loss %f   PSNR_noisy: %f   PSRN_gt: %f PSNR_gt_sm: %f' % (
                i, total_loss.item(), psnr_noisy, psnr_gt, psnr_gt_sm))

            out_np = out.detach().cpu().permute(0, 2, 3, 1).numpy()[0]
            out_sm_np = out_avg.detach().cpu().permute(0, 2, 3, 1).numpy()[0]
            plot_denoising_results(np.array(img_pil), np.array(img_noisy_pil),
                                   out_np, out_sm_np, psnr_gt, psnr_gt_sm, i, EXP, EXP, d['save_dir'])
            plot_training_curves(mse_vals, psnr_gt_vals, psnr_noisy_gt_vals, d['save_dir'])

        # Backtracking
        if i % show_every == 0:
            if psnr_noisy - psnr_noisy_last < -1.5:
                logger.info('Falling back to previous checkpoint.')

                for new_param, net_param in zip(last_net, net.parameters()):
                    net_param.data.copy_(new_param.cuda())

                return total_loss * 0
            else:
                last_net = [x.detach().cpu() for x in net.parameters()]
                psnr_noisy_last = psnr_noisy
                psnr_gt_last = psnr_gt

        i += 1

        return total_loss


    p = get_params(OPT_OVER, net, net_input)
    optimize(OPTIMIZER, p, closure, LR, num_iter, WD)
    plot_training_curves(mse_vals, psnr_gt_vals, psnr_noisy_gt_vals, d['save_dir'])
    out_np = torch_to_np(net(net_input))
