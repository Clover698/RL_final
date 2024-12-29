import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding
import numpy as np
import torch
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
import gc
# from save_img import save_image
class DiffusionEnv(gym.Env):
    def __init__(self, runner, target_steps=10, max_steps=100, worker_id=0):
        super(DiffusionEnv, self).__init__()

        self.runner = copy.deepcopy(runner)
        model, cls = self.runner.get_model()
        self.final_threshold = 0.9
        self.worker_id = worker_id

        # Model
        self.last_T = 999
        self.target_steps = target_steps
        _, _, sigma_0, config, deg, H_funcs, model, idx_so_far, cls_fn = self.runner.sample(
            cls,
            worker_id=worker_id)
        # self.train_loader = train_loader
        self.sigma_0 = sigma_0
        self.config = config
        self.deg = deg
        self.H_funcs = H_funcs
        self.model = model
        self.model.to("cuda")

        self.idx_so_far = idx_so_far
        self.cls_fn = cls_fn
        self.current_image_idx = 0
        self.sample_size = config.data.image_size

        # RL Setting
        self.target_steps = target_steps

        self.max_steps = max_steps

        # Count the number of steps
        self.current_step_num = 0 
        self.action_space = gym.spaces.Box(low=-1, high=1)
        self.uniform_steps = [i for i in range(0, 999, 1000//self.target_steps)][::-1]    
        # else:
        #     self.action_space = spaces.Discrete(20)

        # Define the action and observation space
        self.observation_space = Dict({
            "image": Box(low=-1, high=1, shape=(3, self.sample_size, self.sample_size), dtype=np.float32),
            "value": Box(low=np.array([0]), high=np.array([999]), dtype=np.uint16)
        })

        # ddim sequence
        skip = self.runner.num_timesteps // self.runner.args.timesteps
        self.interval = self.runner.num_timesteps // target_steps
        # seq = range(self.runner.num_timesteps, 0, -1*skip)
        seq = range(0, self.runner.num_timesteps, skip)
        seq_next = [-1] + list(seq[:-1])

        self.ddim_seq = list(reversed(seq))
        self.ddim_seq_next = list(reversed(seq_next))

        self.time_step_sequence = []
        self.action_sequence = []
        self.max_steps = max_steps
        self.current_step_num = 0

        # Initialize the random seed
        self.seed(232 + worker_id)
        self.episode_init = True
        self.state = None
        # pdb.set_trace()
        del runner
        torch.cuda.empty_cache()
        # self.reset()

    def seed(self, seed=None):
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        torch.random.manual_seed(seed)

    def reset(self, seed=None, options=None):
        # print("reset!")
        torch.cuda.empty_cache()
        gc.collect()
        self.episode_init = True
        if seed is not None:
            self.seed(seed + self.worker_id)

        # Reset counter, sequence
        self.current_step_num = 0
        self.time_step_sequence = []
        self.action_sequence = []
        self.data_idx = random.randint(0, len(self.runner.train_dataset)-1)
        self.GT_image, self.classes = self.runner.train_dataset[self.data_idx]
        self.classes = None
        if self.GT_image.dim() == 3:
            self.GT_image = self.GT_image.unsqueeze(0)
        # print("self.GT_image shape:", self.GT_image.shape)

        # # Load Image
        # if self.episode_init or not hasattr(self, 'data_iter'):
        #     self.data_iter = iter(self.train_loader)
        # self.GT_image, self.classes = next(self.data_iter)

        # noise and low level image y_0,
        # x, y_0, pinv_y_0, x_orig, H_inv_y
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
        self.t = self.ddim_seq[0]
        self.x0_t, self.at, self.et = denoise_single_step(self.state, self.model, self.t, self.cls_fn, self.classes)
        self.x0_t = self.pinv_y_0.clone()

        self.ddim_state = initialize_generalized_steps(
            self.pinv_y_0.to("cuda"),
            self.ddim_seq[0],
            self.runner.betas,
            self.H_funcs,
            self.y_0,
            self.sigma_0,
        )
        ddim_x0_t = self.pinv_y_0.clone()

        # Precomputing DDRM uniform seq
        with torch.no_grad():
            for i in range(self.target_steps):
                ddim_t = torch.tensor(self.uniform_steps[i])
                if i != 0:
                    self.ddim_state['x'] = denoise_guided_addnoise(self.ddim_state, ddim_t, ddim_at, ddim_et, ddim_x0_t, self.H_funcs, self.sigma_0, self.runner.args)
                ddim_x0_t, ddim_at, ddim_et = denoise_single_step(self.ddim_state, self.model, ddim_t, self.cls_fn, self.classes)
                # save_image(ddim_x0_t, output_path="./results", file_name="ddim_x0_t.png")
        
        
        orig = inverse_data_transform(self.config, self.GT_image[0]).to(self.runner.device)
        ddim_x = inverse_data_transform(self.config, ddim_x0_t[0]).to(self.runner.device)
        ddim_mse = torch.mean((ddim_x - orig) ** 2)
        self.ddim_psnr = 10 * torch.log10(1 / ddim_mse).item()
        self.ddim_ssim = structural_similarity(
            ddim_x.squeeze(0).cpu().numpy(),
            orig.squeeze(0).cpu().numpy(),
            win_size=21,
            channel_axis=0,
            data_range=1.0
        )
        # save_image(ddim_x, output_path="./results", file_name="ddim_x.png")
        # save_image(inverse_data_transform(self.config, self.GT_image).to(self.runner.device), output_path="./results", file_name="self.GT_image.png")
        # save_image(inverse_data_transform(self.config, self.pinv_y_0).to(self.runner.device), output_path="./results", file_name="self.pinv_y_0.png")
        del ddim_x, ddim_x0_t, ddim_mse, orig
        gc.collect()
        observation = {
            "image":  self.x0_t[0].cpu(),  
            "value": np.array([self.last_T])
        }

        torch.cuda.empty_cache()  # Clear GPU cache

        return observation, {}

    def step(self, action):
        # print("step!!!")
        truncate = self.current_step_num >= self.max_steps
        torch.cuda.empty_cache()
        # RL
        with torch.no_grad():
            # if self.current_step_num == 0:
            # print("interval = ", self.interval)
            start_t = self.t - self.interval - self.interval * action
            # print("start_t = ", start_t)
            # print("current_step_num = ", self.current_step_num)
            # print("first item = ", self.interval*(5-self.current_step_num))
            next_t = torch.tensor(int(max(0, min(start_t, 999))))
            # self.interval = int(next_t / (self.target_steps - 1)) 
            self.state['x'] = denoise_guided_addnoise(self.state, next_t, self.at, self.et, self.x0_t, self.H_funcs, self.sigma_0, self.runner.args)
            self.action_sequence.append(action.item())
            # else:
            #     # next_t = self.t - self.interval - self.interval * action
            #     next_t = int(self.ddim_seq[self.current_step_num] - self.interval * action)
            #     next_t = torch.tensor(int(max(0, min(next_t, 999))))
            #     # self.interval = int(next_t / (self.target_steps - self.current_step_num - 1)) if (self.target_steps - self.current_step_num - 1) != 0 else self.interval
            #     self.state['x'] = denoise_guided_addnoise(self.state, next_t, self.at, self.et, self.x0_t, self.H_funcs, self.sigma_0, self.runner.args)
            #     self.action_sequence.append(action.item())

        self.t = next_t
        self.x0_t, self.at, self.et = denoise_single_step(self.state, self.model, self.t, self.cls_fn, self.classes)
        self.time_step_sequence.append(self.t.item())

        # Finish the episode if denoising is done
        done = (self.current_step_num == self.target_steps - 1)
        reward, ssim, psnr, ddim_ssim, ddim_psnr = self.calculate_reward(done)

        info = {
            'ddim_t': self.uniform_steps[self.current_step_num],
            't': self.t,
            'reward': reward,
            'ssim': ssim,
            'psnr': psnr,
            'ddim_ssim': ddim_ssim,
            'ddim_psnr': ddim_psnr,
            'time_step_sequence': self.time_step_sequence,
            'action_sequence': self.action_sequence,
        }

        observation = {
            "image":  self.x0_t[0].cpu(),  
            "value": np.array([self.t])
        }
        if done:
            del self.x0_t, self.state
            gc.collect()
        self.current_step_num += 1
        torch.cuda.empty_cache()
        return observation, reward, done, truncate, info

    def _perform_denoising_single_step(self, state, t, next_t):
        with torch.no_grad():
            x0_t, xt_next = denoise_single_step(
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
        return x0_t

    def calculate_reward(self, done):
        reward = 0
        orig = inverse_data_transform(self.config, self.GT_image[0]).to(self.runner.device)
        x = inverse_data_transform(self.config, self.x0_t[0]).to(self.runner.device)
        mse = torch.mean((x - orig) ** 2)
        psnr = 10 * torch.log10(1 / mse).item()
        ssim = structural_similarity(x.cpu().numpy(), orig.cpu().numpy(), win_size=21, channel_axis=0, data_range=1.0)
        # ssim = structural_similarity(
        #     x.squeeze(0).cpu().numpy(),
        #     orig.squeeze(0).cpu().numpy(),
        #     win_size=21,
        #     channel_axis=0,
        #     data_range=1.0
        # )        
        # Intermediate reward (Percentage of temporary improvement)
        if ssim > self.ddim_ssim:
            # reward += 0.5/self.target_steps*psnr/self.ddim_psnr 
            reward += 1/self.target_steps

        # Sparse reward (Percentage of final improvement)
        # if done and psnr > self.ddim_psnr and ssim > self.final_threshold: 
        if done and ssim > self.final_threshold:
            # reward += 0.5*psnr/self.ddim_psnr
            # reward += 0.5*ssim/self.ddim_ssim
            reward += 1

        # save_image(x, output_path="./results", file_name="x.png")
        # save_image(orig, output_path="./results", file_name="orig.png")
        # if done:
        #     save_image(x, output_path="./results", file_name="x.png")
        #     save_image(orig, output_path="./results", file_name="orig.png")

        # print("psnr = ", psnr)
        # print("ssim = ", ssim)


        return reward, ssim, psnr, self.ddim_ssim, self.ddim_psnr

    def render(self, mode="human", close=False):
        # This could visualize the current state if necessary
        pass