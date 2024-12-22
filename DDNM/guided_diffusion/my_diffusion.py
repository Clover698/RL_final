import os
import logging
import time
import glob

import numpy as np
import tqdm
import torch
import torch.utils.data as data

from datasets import get_dataset, data_transform, inverse_data_transform
from functions.ckpt_util import get_ckpt_path, download
from functions.svd_ddnm import ddnm_diffusion, ddnm_plus_diffusion

import torchvision.utils as tvu

from guided_diffusion.models import Model
from guided_diffusion.script_util import create_model, create_classifier, classifier_defaults, args_to_dict
import random

from scipy.linalg import orth
from skimage.metrics import structural_similarity as ssim

class_num = 951

def get_gaussian_noisy_img(img, noise_level):
    return img + torch.randn_like(img).cuda() * noise_level

def MeanUpsample(x, scale):
    n, c, h, w = x.shape
    out = torch.zeros(n, c, h, scale, w, scale).to(x.device) + x.view(n,c,h,1,w,1)
    out = out.view(n, c, scale*h, scale*w)
    return out

def color2gray(x):
    coef=1/3
    x = x[:,0,:,:] * coef + x[:,1,:,:]*coef +  x[:,2,:,:]*coef
    return x.repeat(1,3,1,1)

def gray2color(x):
    x = x[:,0,:,:]
    coef=1/3
    base = coef**2 + coef**2 + coef**2
    return torch.stack((x*coef/base, x*coef/base, x*coef/base), 1)    



def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (
            np.linspace(
                beta_start ** 0.5,
                beta_end ** 0.5,
                num_diffusion_timesteps,
                dtype=np.float64,
            )
            ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  
        betas = 1.0 / np.linspace(
            num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


class Diffusion(object):
    def __init__(self, args, config, device=None):
        self.args = args
        self.config = config
        if device is None:
            device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        self.device = device

        self.model_var_type = config.model.var_type
        betas = get_beta_schedule(
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )
        betas = self.betas = torch.from_numpy(betas).float().to(self.device)
        self.num_timesteps = betas.shape[0]

        alphas = 1.0 - betas
        alphas_cumprod = alphas.cumprod(dim=0)
        alphas_cumprod_prev = torch.cat(
            [torch.ones(1).to(device), alphas_cumprod[:-1]], dim=0
        )
        self.alphas_cumprod_prev = alphas_cumprod_prev
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        if self.model_var_type == "fixedlarge":
            self.logvar = betas.log()
        elif self.model_var_type == "fixedsmall":
            self.logvar = posterior_variance.clamp(min=1e-20).log()

        # def sample(self, simplified):
        self.cls_fn = None
        if self.config.model.type == 'simple':
            self.model = Model(self.config)

            if self.config.data.dataset == "CIFAR10":
                name = "cifar10"
            elif self.config.data.dataset == "LSUN":
                name = f"lsun_{self.config.data.category}"
            elif self.config.data.dataset == 'CelebA_HQ':
                name = 'celeba_hq'
            else:
                raise ValueError
            if name != 'celeba_hq':
                ckpt = get_ckpt_path(f"ema_{name}", prefix=self.args.exp)
                print("Loading checkpoint {}".format(ckpt))
            elif name == 'celeba_hq':
                ckpt = os.path.join(self.args.exp, "logs/celeba/celeba_hq.ckpt")
                if not os.path.exists(ckpt):
                    download('https://image-editing-test-12345.s3-us-west-2.amazonaws.com/checkpoints/celeba_hq.ckpt',
                             ckpt)
            else:
                raise ValueError
            self.model.load_state_dict(torch.load(ckpt, map_location=self.device))
            self.model.to(self.device)
            self.model = torch.nn.DataParallel(self.model)

        elif self.config.model.type == 'openai':
            config_dict = vars(self.config.model)
            self.model = create_model(**config_dict)
            if self.config.model.use_fp16:
                self.model.convert_to_fp16()
            if self.config.model.class_cond:
                ckpt = os.path.join(self.args.exp, 'logs/imagenet/%dx%d_diffusion.pt' % (
                self.config.data.image_size, self.config.data.image_size))
                if not os.path.exists(ckpt):
                    download(
                        'https://openaipublic.blob.core.windows.net/diffusion/jul-2021/%dx%d_diffusion_uncond.pt' % (
                        self.config.data.image_size, self.config.data.image_size), ckpt)
            else:
                ckpt = os.path.join(self.args.exp, "logs/imagenet/256x256_diffusion_uncond.pt")
                if not os.path.exists(ckpt):
                    download(
                        'https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt',
                        ckpt)

            self.model.load_state_dict(torch.load(ckpt, map_location=self.device))
            self.model.to(self.device)
            self.model.eval()
            self.model = torch.nn.DataParallel(self.model)

            if self.config.model.class_cond:
                ckpt = os.path.join(self.args.exp, 'logs/imagenet/%dx%d_classifier.pt' % (
                self.config.data.image_size, self.config.data.image_size))
                if not os.path.exists(ckpt):
                    image_size = self.config.data.image_size
                    download(
                        'https://openaipublic.blob.core.windows.net/diffusion/jul-2021/%dx%d_classifier.pt' % image_size,
                        ckpt)
                self.classifier = create_classifier(**args_to_dict(self.config.classifier, classifier_defaults().keys()))
                self.classifier.load_state_dict(torch.load(ckpt, map_location=self.device))
                self.classifier.to(self.device)
                if self.config.classifier.classifier_use_fp16:
                    self.classifier.convert_to_fp16()
                self.classifier.eval()
                self.classifier = torch.nn.DataParallel(self.classifier)

                import torch.nn.functional as F
                def cond_fn(x, t, y):
                    with torch.enable_grad():
                        x_in = x.detach().requires_grad_(True)
                        logits = classifier(x_in, t)
                        log_probs = F.log_softmax(logits, dim=-1)
                        selected = log_probs[range(len(logits)), y.view(-1)]
                        return torch.autograd.grad(selected.sum(), x_in)[0] * self.config.classifier.classifier_scale

                self.cls_fn = cond_fn

        # if simplified:
        #     print('Run Simplified DDNM, without SVD.',
        #           f'{self.config.time_travel.T_sampling} sampling steps.',
        #           f'travel_length = {self.config.time_travel.travel_length},',
        #           f'travel_repeat = {self.config.time_travel.travel_repeat}.',
        #           f'Task: {self.args.deg}.'
        #          )
        #     self.simplified_ddnm_plus(model, cls_fn)
        # else:
        #     print('Run SVD-based DDNM.',
        #           f'{self.config.time_travel.T_sampling} sampling steps.',
        #           f'travel_length = {self.config.time_travel.travel_length},',
        #           f'travel_repeat = {self.config.time_travel.travel_repeat}.',
        #           f'Task: {self.args.deg}.'
        #          )
        #     self.svd_based_ddnm_plus(model, cls_fn)

        args, config = self.args, self.config

        _, self.dataset, self.test_dataset = get_dataset(args, config)

        device_count = torch.cuda.device_count()

        if args.subset_start >= 0 and args.subset_end > 0:
            assert args.subset_end > args.subset_start
            self.test_dataset = torch.utils.data.Subset(self.test_dataset, range(args.subset_start, args.subset_end))
        else:
            args.subset_start = 0
            args.subset_end = len(self.test_dataset)

        print(f'Dataset has size {len(self.test_dataset)}')

        def seed_worker(worker_id):
            worker_seed = args.seed % 2 ** 32
            np.random.seed(worker_seed)
            random.seed(worker_seed)

        # g = torch.Generator()
        # g.manual_seed(args.seed)
        # self.val_loader = data.DataLoader(
        #     test_dataset,
        #     batch_size=config.sampling.batch_size,
        #     shuffle=True,
        #     num_workers=config.data.num_workers,
        #     worker_init_fn=seed_worker,
        #     generator=g,
        # )

        # get degradation matrix
        self.deg = args.deg
        self.A_funcs = None
        if self.deg == 'cs_walshhadamard':
            compress_by = round(1/args.deg_scale)
            from functions.svd_operators import WalshHadamardCS
            self.A_funcs = WalshHadamardCS(config.data.channels, self.config.data.image_size, compress_by,
                                      torch.randperm(self.config.data.image_size ** 2, device=self.device), self.device)
        elif self.deg == 'cs_blockbased':
            cs_ratio = args.deg_scale
            from functions.svd_operators import CS
            self.A_funcs = CS(config.data.channels, self.config.data.image_size, cs_ratio, self.device)
        elif self.deg == 'inpainting':
            from functions.svd_operators import Inpainting
            loaded = np.load("exp/inp_masks/mask.npy")
            mask = torch.from_numpy(loaded).to(self.device).reshape(-1)
            missing_r = torch.nonzero(mask == 0).long().reshape(-1) * 3
            missing_g = missing_r + 1
            missing_b = missing_g + 1
            missing = torch.cat([missing_r, missing_g, missing_b], dim=0)
            self.A_funcs = Inpainting(config.data.channels, config.data.image_size, missing, self.device)
        elif self.deg == 'denoising':
            from functions.svd_operators import Denoising
            self.A_funcs = Denoising(config.data.channels, self.config.data.image_size, self.device)
        elif self.deg == 'colorization':
            from functions.svd_operators import Colorization
            self.A_funcs = Colorization(config.data.image_size, self.device)
        elif self.deg == 'sr_averagepooling':
            blur_by = int(args.deg_scale)
            from functions.svd_operators import SuperResolution
            self.A_funcs = SuperResolution(config.data.channels, config.data.image_size, blur_by, self.device)
        elif self.deg == 'sr_bicubic':
            factor = int(args.deg_scale)
            from functions.svd_operators import SRConv
            def bicubic_kernel(x, a=-0.5):
                if abs(x) <= 1:
                    return (a + 2) * abs(x) ** 3 - (a + 3) * abs(x) ** 2 + 1
                elif 1 < abs(x) and abs(x) < 2:
                    return a * abs(x) ** 3 - 5 * a * abs(x) ** 2 + 8 * a * abs(x) - 4 * a
                else:
                    return 0
            k = np.zeros((factor * 4))
            for i in range(factor * 4):
                x = (1 / factor) * (i - np.floor(factor * 4 / 2) + 0.5)
                k[i] = bicubic_kernel(x)
            k = k / np.sum(k)
            kernel = torch.from_numpy(k).float().to(self.device)
            self.A_funcs = SRConv(kernel / kernel.sum(), \
                             config.data.channels, self.config.data.image_size, self.device, stride=factor)
        elif self.deg == 'deblur_uni':
            from functions.svd_operators import Deblurring
            self.A_funcs = Deblurring(torch.Tensor([1 / 9] * 9).to(self.device), config.data.channels,
                                 self.config.data.image_size, self.device)
        elif self.deg == 'deblur_gauss':
            from functions.svd_operators import Deblurring
            sigma = 10
            pdf = lambda x: torch.exp(torch.Tensor([-0.5 * (x / sigma) ** 2]))
            kernel = torch.Tensor([pdf(-2), pdf(-1), pdf(0), pdf(1), pdf(2)]).to(self.device)
            self.A_funcs = Deblurring(kernel / kernel.sum(), config.data.channels, self.config.data.image_size, self.device)
        elif self.deg == 'deblur_aniso':
            from functions.svd_operators import Deblurring2D
            sigma = 20
            pdf = lambda x: torch.exp(torch.Tensor([-0.5 * (x / sigma) ** 2]))
            kernel2 = torch.Tensor([pdf(-4), pdf(-3), pdf(-2), pdf(-1), pdf(0), pdf(1), pdf(2), pdf(3), pdf(4)]).to(
                self.device)
            sigma = 1
            pdf = lambda x: torch.exp(torch.Tensor([-0.5 * (x / sigma) ** 2]))
            kernel1 = torch.Tensor([pdf(-4), pdf(-3), pdf(-2), pdf(-1), pdf(0), pdf(1), pdf(2), pdf(3), pdf(4)]).to(
                self.device)
            self.A_funcs = Deblurring2D(kernel1 / kernel1.sum(), kernel2 / kernel2.sum(), config.data.channels,
                                   self.config.data.image_size, self.device)
        else:
            raise ValueError("degradation type not supported")
        args.sigma_y = 2 * args.sigma_y #to account for scaling to [-1,1]
        self.sigma_y = args.sigma_y

    def preprocess(self, x_orig, idx_so_far):        
        args, config = self.args, self.config
        x_orig = x_orig.unsqueeze(0).to(self.device)
        x_orig = data_transform(self.config, x_orig)

        y = self.A_funcs.A(x_orig)
            
        b, hwc = y.size()
        if 'color' in self.deg:
            hw = hwc / 1
            h = w = int(hw ** 0.5)
            y = y.reshape((b, 1, h, w))
        elif 'inp' in self.deg or 'cs' in self.deg:
            pass
        else:
            hw = hwc / 3
            h = w = int(hw ** 0.5)
            y = y.reshape((b, 3, h, w))
            
        if self.args.add_noise: # for denoising test
            y = get_gaussian_noisy_img(y, self.sigma_y) 
        
        y = y.reshape((b, hwc))
        
        Apy = self.A_funcs.A_pinv(y).view(y.shape[0], config.data.channels, self.config.data.image_size,
                                            self.config.data.image_size)

        if self.deg[:6] == 'deblur':
            Apy = y.view(y.shape[0], config.data.channels, self.config.data.image_size,
                                self.config.data.image_size)
        elif self.deg == 'colorization':
            Apy = y.view(y.shape[0], 1, self.config.data.image_size, self.config.data.image_size).repeat(1,3,1,1)
        elif self.deg == 'inpainting':
            Apy += self.A_funcs.A_pinv(self.A_funcs.A(torch.ones_like(Apy))).reshape(*Apy.shape) - 1

        # os.makedirs(os.path.join(self.args.image_folder, "Apy"), exist_ok=True)
        # for i in range(len(Apy)):
        #     tvu.save_image(
        #         inverse_data_transform(config, Apy[i]),
        #         os.path.join(self.args.image_folder, f"Apy/Apy_{idx_so_far + i}.png")
        #     )
        #     tvu.save_image(
        #         inverse_data_transform(config, x_orig[i]),
        #         os.path.join(self.args.image_folder, f"Apy/orig_{idx_so_far + i}.png")
        #     )

        #Start DDIM
        x = torch.randn(
            y.shape[0],
            config.data.channels,
            config.data.image_size,
            config.data.image_size,
            device=self.device,
        )
        A_inv_y = self.A_funcs.A_pinv(y.reshape(y.size(0), -1)).reshape(x.size())
        return x, y, Apy, x_orig, A_inv_y

    def postprocess(self, x, x_orig, idx_so_far):
        x = [inverse_data_transform(self.config, x)]


        tvu.save_image(
            x[0], os.path.join(self.args.image_folder, f"{idx_so_far}_{0}.png")
        )
        #     orig = inverse_data_transform(self.config, x_orig[j])
        #     mse = torch.mean((x[0][j].to(self.device) - orig) ** 2)
        #     psnr = 10 * torch.log10(1 / mse)
        #     per_ssim = ssim(np.transpose(x[0][j].cpu().numpy(), (1,2,0)), np.transpose(orig.cpu().numpy(), (1,2,0)), win_size=21, multichannel=True, data_range=1.0)
        # return psnr, per_ssim
    
    def single_step_ddnm(self, x, y, i, classes):
        xt = x.to('cuda')
        t = torch.tensor([i]).to(xt.device)
        at = compute_alpha(self.betas, t.long())
        if self.cls_fn == None:
            et = self.model(xt, t)
        else:
            classes = torch.ones(xt.size(0), dtype=torch.long, device=torch.device("cuda"))*class_num
            et = self.model(xt, t, classes)
            et = et[:, :3]
            et = et - (1 - at).sqrt()[0, 0, 0, 0] * self.cls_fn(x, t, classes)

        if et.size(1) == 6:
            et = et[:, :3]

        x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()

        x0_t_hat = x0_t - self.A_funcs.A_pinv(
            self.A_funcs.A(x0_t.reshape(x0_t.size(0), -1)) - y.reshape(y.size(0), -1)
        ).reshape(*x0_t.size())

        return x0_t_hat.cpu(), x0_t.cpu(), et.cpu()
        # return x0_t_hat, x0_t, et
    
    def get_noisy_x(self, next_t, x0_t_hat, et=None, initial=False):
        next_t = next_t.to('cuda')
        x0_t_hat = x0_t_hat.to('cuda')
        at_next = compute_alpha(self.betas, next_t.long())
        if initial:
            xt_next = at_next.sqrt() * x0_t_hat + torch.randn_like(x0_t_hat) * (1 - at_next).sqrt()
        else:
            c1 = (1 - at_next).sqrt() * self.args.eta
            c2 = (1 - at_next).sqrt() * ((1 - self.args.eta ** 2) ** 0.5)
            xt_next = at_next.sqrt() * x0_t_hat + c1 * torch.randn_like(x0_t_hat) + c2 * et.to('cuda')
        return xt_next
    
# Code form RePaint   
def get_schedule_jump(T_sampling, travel_length, travel_repeat):
    jumps = {}
    for j in range(0, T_sampling - travel_length, travel_length):
        jumps[j] = travel_repeat - 1

    t = T_sampling
    ts = []

    while t >= 1:
        t = t-1
        ts.append(t)

        if jumps.get(t, 0) > 0:
            jumps[t] = jumps[t] - 1
            for _ in range(travel_length):
                t = t + 1
                ts.append(t)

    ts.append(-1)

    _check_times(ts, -1, T_sampling)
    return ts

def _check_times(times, t_0, T_sampling):
    # Check end
    assert times[0] > times[1], (times[0], times[1])

    # Check beginning
    assert times[-1] == -1, times[-1]

    # Steplength = 1
    for t_last, t_cur in zip(times[:-1], times[1:]):
        assert abs(t_last - t_cur) == 1, (t_last, t_cur)

    # Value range
    for t in times:
        assert t >= t_0, (t, t_0)
        assert t <= T_sampling, (t, T_sampling)
        
def compute_alpha(beta, t):
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
    return a
