import torch
import torchvision
import os

from PIL import Image, ImageFont, ImageDraw
from skimage.metrics import structural_similarity as ssim
import numpy as np
import matplotlib.pyplot as plt


def crop_image(img, d=32):
    '''Make dimensions divisible by `d`'''

    new_size = (img.size[0] - img.size[0] % d, 
                img.size[1] - img.size[1] % d)

    bbox = [
            int((img.size[0] - new_size[0])/2), 
            int((img.size[1] - new_size[1])/2),
            int((img.size[0] + new_size[0])/2),
            int((img.size[1] + new_size[1])/2),
    ]

    img_cropped = img.crop(bbox)
    return img_cropped


def get_params(opt_over, net, net_input, downsampler=None):
    '''Returns parameters that we want to optimize over.

    Args:
        opt_over: comma separated list, e.g. "net,input" or "net"
        net: network
        net_input: torch.Tensor that stores input `z`
    '''
    opt_over_list = opt_over.split(',')
    params = []
    
    for opt in opt_over_list:
    
        if opt == 'net':
            params += [x for x in net.parameters() ]
        elif  opt=='down':
            assert downsampler is not None
            params = [x for x in downsampler.parameters()]
        elif opt == 'input':
            net_input.requires_grad = True
            params += [net_input]
        else:
            assert False, 'what is it?'
            
    return params


def get_image_grid(images_np, nrow=8):
    '''Creates a grid from a list of images by concatenating them.'''
    images_torch = [torch.from_numpy(x) for x in images_np]
    torch_grid = torchvision.utils.make_grid(images_torch, nrow)
    
    return torch_grid.numpy()


def plot_image_grid(images_np, count, nrow =8, factor=1, interpolation='lanczos',):
    """Draws images in a grid
    
    Args:
        images_np: list of images, each image is np.array of size 3xHxW of 1xHxW
        nrow: how many images will be in one row
        factor: size if the plt.figure 
        interpolation: interpolation used in plt.imshow
    """
    n_channels = max(x.shape[0] for x in images_np)
    assert (n_channels == 3) or (n_channels == 1), "images should have 1 or 3 channels"
    
    images_np = [x if (x.shape[0] == n_channels) else np.concatenate([x, x, x], axis=0) for x in images_np]

    grid = get_image_grid(images_np, nrow)
    
    plt.figure(figsize=(len(images_np) + factor, 12 + factor))
    
    if images_np[0].shape[0] == 1:
        plt.imshow(grid[0], cmap='gray', interpolation=interpolation)
    else:
        plt.imshow(grid.transpose(1, 2, 0), interpolation=interpolation)

    plt.axis('off')
    plt.title('Transformer last layer - iter #{}'.format(count))
    plt.savefig('transformer_last_layer_{}'.format(count))
    plt.show()
    
    return grid


def load(path):
    """Load PIL image."""
    img = Image.open(path)
    return img


def get_image(path, imsize=-1):
    """Load an image and resize to a cpecific size. 

    Args: 
        path: path to image
        imsize: tuple or scalar with dimensions; -1 for `no resize`
    """
    img = load(path)

    if isinstance(imsize, int):
        imsize = (imsize, imsize)

    if imsize[0]!= -1 and img.size != imsize:
        if imsize[0] > img.size[0]:
            img = img.resize(imsize, Image.BICUBIC)
        else:
            img = img.resize(imsize, Image.ANTIALIAS)

    img_np = pil_to_np(img)

    return img, img_np


def fill_noise(x, noise_type):
    """Fills tensor `x` with noise of type `noise_type`."""
    if noise_type == 'u':
        x.uniform_()
    elif noise_type == 'n':
        x.normal_() 
    else:
        assert False


def get_noise(input_depth, method, spatial_size, noise_type='u', var=1./10):
    """Returns a pytorch.Tensor of size (1 x `input_depth` x `spatial_size[0]` x `spatial_size[1]`) 
    initialized in a specific way.
    Args:
        input_depth: number of channels in the tensor
        method: `noise` for fillting tensor with noise; `meshgrid` for np.meshgrid
        spatial_size: spatial size of the tensor to initialize
        noise_type: 'u' for uniform; 'n' for normal
        var: a factor, a noise will be multiplicated by. Basically it is standard deviation scaler. 
    """
    if isinstance(spatial_size, int):
        spatial_size = (spatial_size, spatial_size)
    if method == 'noise':
        shape = [1, input_depth, spatial_size[0], spatial_size[1]]
        net_input = torch.zeros(shape)
        
        fill_noise(net_input, noise_type)
        net_input *= var            
    elif method == 'meshgrid': 
        assert input_depth == 2
        X, Y = np.meshgrid(np.arange(0, spatial_size[1])/float(spatial_size[1]-1), np.arange(0, spatial_size[0])/float(spatial_size[0]-1))
        meshgrid = np.concatenate([X[None,:], Y[None,:]])
        net_input=  np_to_torch(meshgrid)
    else:
        assert False
        
    return net_input


def pil_to_np(img_PIL):
    '''Converts image in PIL format to np.array.
    
    From W x H x C [0...255] to C x W x H [0..1]
    '''
    ar = np.array(img_PIL)

    if len(ar.shape) == 3:
        ar = ar.transpose(2,0,1)
    else:
        ar = ar[None, ...]

    return ar.astype(np.float32) / 255.


def np_to_pil(img_np): 
    '''Converts image in np.array format to PIL image.
    
    From C x W x H [0..1] to  W x H x C [0...255]
    '''
    ar = np.clip(img_np*255,0,255).astype(np.uint8)
    
    if img_np.shape[0] == 1:
        ar = ar[0]
    else:
        ar = ar.transpose(1, 2, 0)

    return Image.fromarray(ar)


def np_to_torch(img_np):
    '''Converts image in numpy.array to torch.Tensor.

    From C x W x H [0..1] to  C x W x H [0..1]
    '''
    return torch.from_numpy(img_np)[None, :]


def torch_to_np(img_var):
    '''Converts an image in torch.Tensor format to np.array.

    From 1 x C x W x H [0..1] to  C x W x H [0..1]
    '''
    return img_var.detach().cpu().numpy()[0]


def optimize(optimizer_type, parameters, closure, LR, num_iter, WD):
    """Runs optimization loop.

    Args:
        optimizer_type: 'LBFGS' of 'adam'
        parameters: list of Tensors to optimize over
        closure: function, that returns loss variable
        LR: learning rate
        num_iter: number of iterations 
    """
    if optimizer_type == 'LBFGS':
        # Do several steps with adam first
        optimizer = torch.optim.Adam(parameters, lr=0.001)
        for j in range(100):
            optimizer.zero_grad()
            closure()
            optimizer.step()

        print('Starting optimization with LBFGS')        
        def closure2():
            optimizer.zero_grad()
            return closure()
        optimizer = torch.optim.LBFGS(parameters, max_iter=num_iter, lr=LR, tolerance_grad=-1, tolerance_change=-1)
        optimizer.step(closure2)

    elif optimizer_type == 'adam':
        print('Starting optimization with ADAM')
        optimizer = torch.optim.Adam(parameters, lr=LR)
        
        for j in range(num_iter):
            optimizer.zero_grad()
            closure()
            optimizer.step()

    elif optimizer_type == 'adamW':
        print('Starting optimization with ADAM-W')
        optimizer = torch.optim.AdamW(parameters, lr=LR, weight_decay=WD)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_iter)
        for j in range(num_iter):
            optimizer.zero_grad()
            closure()
            optimizer.step()
            # scheduler.step()
    else:
        assert False


best_psnr_gt = -1


def plot_denoising_results(
        img_org, img_noise,
        current_res,
        psnr_gt,
        count, filename, save_dir):

    global best_psnr_gt
    fig, axes = plt.subplots(1, 3, figsize=(20, 8))
    axes[0].imshow(img_org)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    axes[1].imshow(img_noise)
    axes[1].set_title('Noisy Image')
    axes[1].axis('off')
    axes[2].imshow(current_res)
    axes[2].set_title('Current Result (#{})\n PSNR {:2.3f}'.format(count, psnr_gt))
    axes[2].axis('off')
    plt.savefig(os.path.join(save_dir, '{}_{}.png'.format(filename, count)))
    plt.close(fig)

    if psnr_gt > best_psnr_gt:
        best_psnr_gt = psnr_gt
        ssim_res = ssim((img_org / 255).astype(np.float32), current_res, multichannel=True,
                        data_range=current_res.max() - current_res.min())
        current_res_uint8 = (current_res * 255).astype(np.uint8)
        img_pil = Image.fromarray(current_res_uint8)
        draw = ImageDraw.Draw(img_pil)
        font = ImageFont.load_default()
        draw.text((0, 0), 'PSNR: {:.3f}'.format(psnr_gt), (0, 0, 0), font=font)
        draw.text((0, 10), 'SSIM: {:.3f}'.format(ssim_res), (0, 0, 0), font=font)
        img_pil.save(os.path.join(save_dir, '{}_best.png'.format(filename)))


def plot_training_curves(loss_vals, eval_vals, noise_eval_vals, save_dir):
    assert len(loss_vals) == len(eval_vals), "loss {} and eval {} lists are not in the same length".format(
        len(loss_vals), len(eval_vals))
    fig, ax = plt.subplots(3, 1, figsize=(20, 20))
    ths = [100, 200]
    stpes_vec = [i for i in range(len(loss_vals))]
    ax[0].plot(stpes_vec, loss_vals)
    ax[0].set_xlabel('steps')
    ax[0].set_ylabel('mse')
    ax[0].set_title('MSE')
    ax[0].vlines(ths, ymin=0, ymax=1, color='r')

    ax[1].plot(stpes_vec, eval_vals)
    ax[1].set_xlabel('steps')
    ax[1].set_ylabel('dB')
    ax[1].set_title('PSNR-GT')
    ax[1].vlines(ths, ymin=0, ymax=1, color='r')

    ax[2].plot(stpes_vec, noise_eval_vals)
    ax[2].set_xlabel('steps')
    ax[2].set_ylabel('dB')
    ax[2].set_title('PSNR-Noisy')
    ax[2].vlines(ths, ymin=0, ymax=1, color='r')

    plt.savefig(os.path.join(save_dir, 'training_curves.png'))
    plt.close(fig)


def plot_grad_flow(named_parameters):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.

    Usage: Plug this function in Trainer class after loss.backwards() as
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    ave_grads = []
    max_grads = []
    layers = []
    for n, p in named_parameters:
        if p.requires_grad and "bias" not in n:
            layers.append(n)
            try:
                ave_grads.append(p.grad.abs().mean().cpu())
                max_grads.append(p.grad.abs().max().cpu())
            except:
                print(n, p)
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads) + 1, lw=2, color="k")
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom=-0.001, top=0.02)  # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([plt.Line2D([0], [0], color="c", lw=4),
                plt.Line2D([0], [0], color="b", lw=4),
                plt.Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])


curr_iter = 0
save_dir = '.'


def get_current_iter_num():
    global curr_iter
    return curr_iter


def set_current_iter_num(val):
    global curr_iter
    curr_iter = val

def set_save_dir(val):
    global save_dir
    save_dir = val


def get_save_dir():
    global save_dir
    return save_dir


def attention_debug_func(attention_map, debug_name):
    b, num_heads, t1, t2 = attention_map.shape
    plot_limit = num_heads // 2
    fig, ax = plt.subplots(2, plot_limit, figsize=(20, 10))
    for attn_ind in range(num_heads):
        current_map = attention_map[0, attn_ind]
        current_ax = ax[attn_ind // plot_limit][attn_ind % plot_limit]
        mean, std = torch.std_mean(current_map[current_map != 0.0])
        # vmin, vmax = torch.min(current_map[current_map != 0.0]), torch.max(current_map[current_map != 0.0])
        im = current_ax.imshow(current_map.detach().cpu().numpy(), cmap='gray')
        # im.clim(vmin, vmax)
        fig.colorbar(im, ax=current_ax, fraction=0.046, pad=0.04)
        current_ax.set_title('Head #{0}\n mean: {1:2.3f} std: {2:2.3f}'.format(attn_ind, mean.detach(), std.detach()))

    plt.tight_layout()
    plt.savefig(os.path.join(get_save_dir(), '{}_iter{}.png'.format(debug_name, get_current_iter_num())))
    plt.close(fig)


