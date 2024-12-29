import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding
import numpy as np
import torch

# from diffusers_old import DDIMPipeline, DDIMScheduler, UNet2DModel
import os
from PIL import Image
from skimage.metrics import structural_similarity
from gymnasium.spaces import Box, Dict
import os
import random

# DDRM
from ddrm.datasets import get_dataset, data_transform, inverse_data_transform
from ddrm.functions.denoising import initialize_generalized_steps, denoise_single_step, denoise_guided_addnoise
import pdb
import copy


class EvalDiffusionEnv(gym.Env):
    def __init__(
        self,
        runner, 
        target_steps=10,
        max_steps=100,
    ):
        super(EvalDiffusionEnv, self).__init__()

        self.img_idx_so_far = 0
             
        # Model
        self.last_T = 999
        self.runner = copy.deepcopy(runner)
        model, cls = self.runner.get_model()
        self.target_steps = target_steps
        self.final_threshold = 0.9
        _, _, sigma_0, config, deg, H_funcs, model, idx_so_far, cls_fn = self.runner.sample(cls)
        # self.val_loader = val_loader
        self.sigma_0 = sigma_0
        self.config = config
        self.deg = deg
        self.H_funcs = H_funcs
        self.model = model
        self.model.to("cuda")
        
        self.idx_so_far = idx_so_far
        self.cls_fn = cls_fn
        # self.valdata_len = self.runner.val_datalen
        self.current_image_idx = 0
        self.sample_size = config.data.image_size
        self.batch_size = config.sampling.batch_size

        # RL Setting
        self.target_steps = target_steps
        self.uniform_steps = [i for i in range(0, 999, 1000//self.target_steps)][::-1]    
        skip = self.runner.num_timesteps // self.runner.args.timesteps
        self.interval = self.runner.num_timesteps // target_steps
        seq = range(0, self.runner.num_timesteps, skip)
        seq_next = [-1] + list(seq[:-1])

        self.ddim_seq = list(reversed(seq))
        self.ddim_seq_next = list(reversed(seq_next))
        self.max_steps = max_steps

        # Count the number of steps
        self.current_step_num = 0 
        self.action_space = gym.spaces.Box(low=-5, high=5)

        # Define the action and observation space
        self.observation_space = Dict({
            "image": Box(low=-1, high=1, shape=(3, self.sample_size, self.sample_size), dtype=np.float32),
            "value": Box(low=np.array([0]), high=np.array([999]), dtype=np.uint16)
        })

    def seed(self, seed=None):
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        torch.random.manual_seed(seed)

    def reset(self, seed=None, options=None):
        self.episode_init = True
        if seed is not None:
            self.seed(seed)

        # Reset counter, sequence
        self.current_step_num = 0
        self.time_step_sequence = []
        self.action_sequence = []

        self.data_idx = random.randint(0, len(self.runner.test_dataset)-1)
        self.GT_image, self.classes = self.runner.test_dataset[self.data_idx]
        if self.GT_image.dim() == 3:
            self.GT_image = self.GT_image.unsqueeze(0)
        # Load Image
        # self.data_iter = iter(self.val_loader)
        # self.GT_image, self.classes = next(self.data_iter) 

        # noise and low level image y_0, 
        self.noise_image, self.y_0, self.pinv_y_0, self.GT_image, self.H_inv_y = (
            self.runner.sample_init(
                self.GT_image,
                self.sigma_0,
                self.config,
                self.deg,
                self.H_funcs,
                self.model,
                self.idx_so_far,
                self.cls_fn,
                self.classes,
            )
        )


        # Initialization, extract degradation information from y_0 sigma 0, and H_func
        self.state = initialize_generalized_steps(
                self.pinv_y_0.to("cuda"),
                self.last_T,
                self.runner.betas,
                self.H_funcs,
                self.y_0,
                self.sigma_0,
            )
        # self.x0_t = self.state['x']
        self.t = self.ddim_seq[0]
        self.x0_t, self.at, self.et = denoise_single_step(self.state, self.model, self.t, self.cls_fn, self.classes)
        self.x0_t = self.pinv_y_0.clone()

        observation = {
            "image": self.x0_t[0].cpu(),  
            "value": np.array([999])
        }

        torch.cuda.empty_cache()  # Clear GPU cache
        return observation, {}


    def step(self, action):
        truncate = self.current_step_num >= self.max_steps

        with torch.no_grad():
            next_t = self.t - self.interval - self.interval * action
            next_t = torch.tensor(int(max(0, min(next_t, 999))))
            self.interval = int(next_t / (self.target_steps - self.current_step_num - 1)) if (self.target_steps - self.current_step_num - 1) != 0 else self.interval
            self.state['x'] = denoise_guided_addnoise(self.state, next_t, self.at, self.et, self.x0_t, self.H_funcs, self.sigma_0, self.runner.args)
            self.action_sequence.append(action.item())

            self.t = next_t
            self.x0_t, self.at, self.et = denoise_single_step(self.state, self.model, self.t, self.cls_fn, self.classes)
            self.time_step_sequence.append(self.t.item())

        # Finish the episode if denoising is done
        done = self.current_step_num == self.target_steps - 1
        # Calculate reward
        reward, ssim, psnr = self.calculate_reward(done)
        
        if done:
            self.runner.save_img(self.x0_t, self.img_idx_so_far)
            self.img_idx_so_far += 1 if self.img_idx_so_far < len(self.runner.test_dataset) - 1 else 0
            
        info = {
            'ddim_t': self.uniform_steps[self.current_step_num],
            't': self.t,
            'reward': reward,
            'ssim': ssim,
            'psnr': psnr,
            'time_step_sequence': self.time_step_sequence,
            'action_sequence': self.action_sequence,
            'threshold': self.final_threshold,
        }

        observation = {
            "image": self.x0_t[0].cpu(),  
            "value": np.array([self.t])
        }

        self.current_step_num += 1

        return observation, reward, done, truncate, info

    def _update_sequences(self, t, action):
        self.time_step_sequence.append(t.item() if type(t) == torch.Tensor else t)
        self.action_sequence.append(action.item())

    def _perform_denoising_single_step(self, state, t, next_t):
        with torch.no_grad():
            xs, x0_preds = denoise_single_step(
                state,
                self.model,
                t,
                next_t,
                self.runner.betas,
                self.H_funcs,
                self.sigma_0,
                etaB=self.runner.args.etaB,
                etaA=self.runner.args.eta,
                etaC=self.runner.args.eta,
                cls_fn=self.cls_fn,
                classes=self.classes,
            )
            x = torch.stack([inverse_data_transform(self.config, y) for y in xs])
        return x

    def calculate_reward(self, done):
        reward = 0
        x = inverse_data_transform(self.config, self.x0_t).to(self.runner.device)
        orig = inverse_data_transform(self.config, self.GT_image).to(self.runner.device)
        mse = torch.mean((x - orig) ** 2)
        psnr = 10 * torch.log10(1 / mse).item()
        # ssim = structural_similarity(x.cpu().numpy(), orig.cpu().numpy(), win_size=21, channel_axis=0, data_range=1.0)
        ssim = structural_similarity(
            x.squeeze(0).cpu().numpy(),
            orig.squeeze(0).cpu().numpy(),
            win_size=21,
            channel_axis=0,
            data_range=1.0
        )
        # Sparse reward (SSIM)
        if done and ssim > self.final_threshold:
            reward += 1

        return reward, ssim, psnr

    def get_sample_number(self):
        return self.sample_number_count

    def render(self, mode="human", close=False):
        # This could visualize the current state if necessary
        pass