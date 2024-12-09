import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding
import numpy as np
import torch
import os
from PIL import Image
from gymnasium.spaces import Box, Dict
import os
import random
from datasets import get_dataset, data_transform, inverse_data_transform
from skimage.metrics import structural_similarity

class EvalDiffusionEnv(gym.Env):
    def __init__(self, target_steps=10, max_steps=100, threshold=0.8, DM=None, agent1=None):
        super(EvalDiffusionEnv, self).__init__()
        self.DM = DM
        self.agent1 = agent1
        self.target_steps = target_steps
        self.uniform_steps = [i for i in range(0, 999, 1000//target_steps)][::-1]
        # Threshold for the sparse reward
        self.final_threshold = threshold
        
        self.sample_size = 256
        # Maximum number of steps  (Baseline)
        self.max_steps = max_steps 
        # Count the number of steps
        self.current_step_num = 0 
        # Define the action and observation space
        self.action_space = gym.spaces.Box(low=-5, high=5)
        self.observation_space = Dict({
            "image": Box(low=-1, high=1, shape=(3, self.sample_size, self.sample_size), dtype=np.float32),
            "value": Box(low=np.array([0]), high=np.array([999]), dtype=np.uint16)
        })
        # Initialize the random seed
        self.seed(232)
        self.data_idx = 0
        self.reset()
        
    def seed(self, seed=None):
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        torch.random.manual_seed(seed)
    
    def reset(self, seed=None, options=None):
        if seed is not None:
            self.seed(seed)
        self.current_step_num = 0
        self.time_step_sequence = []
        self.action_sequence = []
        self.x_orig, self.classes = self.DM.test_dataset[self.data_idx]
        self.x, self.y, self.Apy, self.x_orig, self.A_inv_y = self.DM.preprocess(self.x_orig, self.data_idx)
        self.x0_t = self.A_inv_y.clone()

        observation = {
            "image": self.x0_t[0].cpu(),  
            "value": np.array([999])
        }
        with torch.no_grad():
            action, _state = self.agent1.predict(observation, deterministic=True)
            start_t = 50 * (1+action) - 1
            t = torch.tensor(int(max(0, min(start_t, 999))))
            self.interval = int(t / (self.target_steps - 1)) 
            self.x = self.DM.get_noisy_x(t, self.x0_t, initial=True)
            self.action_sequence.append(action.item())
            self.previous_t = t
            self.x0_t, _,  self.et = self.DM.single_step_ddnm(self.x, self.y, t, self.classes)
            self.time_step_sequence.append(t.item())
            observation = {
                "image": self.x0_t[0].cpu(),
                "value": np.array([t])
            }

        torch.cuda.empty_cache()  # Clear GPU cache
        return observation, {}
    
    def step(self, action):
        truncate = True if self.current_step_num >= self.max_steps else False
        with torch.no_grad():
            t = self.previous_t - self.interval - self.interval * float(action - 100.0) / 200.0
            t = torch.tensor(int(max(0, min(t, 999))))
            self.interval = int(t / (self.target_steps - self.current_step_num - 1)) if (self.target_steps - self.current_step_num - 1) != 0 else self.interval
            self.x = self.DM.get_noisy_x(t, self.x0_t, self.et)
            self.action_sequence.append(action.item())
            self.previous_t = t
            self.x0_t, _,  self.et = self.DM.single_step_ddnm(self.x, self.y, t, self.classes)
            self.time_step_sequence.append(t.item())


        # Finish the episode if denoising is done
        done = self.current_step_num == self.target_steps - 1
        # Calculate reward
        reward, ssim, psnr = self.calculate_reward(done)
        if done:
            self.DM.postprocess(self.x0_t, self.x_orig, self.data_idx)
            self.data_idx += 1 if self.data_idx < len(self.DM.test_dataset) - 1 else 0
        info = {
            'ddim_t': self.uniform_steps[self.current_step_num],
            't': t,
            'reward': reward,
            'ssim': ssim,
            'psnr': psnr,
            'time_step_sequence': self.time_step_sequence,
            'action_sequence': self.action_sequence,
            'threshold': self.final_threshold,
        }
        # print('info:', info)
        observation = {
            "image": self.x0_t[0].cpu(),  
            "value": np.array([t])
        }
        # Increase number of steps
        self.current_step_num += 1
       
        return observation, reward, done, truncate, info

    def calculate_reward(self, done):
        reward = 0
        x = inverse_data_transform(self.DM.config, self.x0_t[0]).to(self.DM.device)
        orig = inverse_data_transform(self.DM.config, self.x_orig[0]).to(self.DM.device)
        mse = torch.mean((x - orig) ** 2)
        psnr = 10 * torch.log10(1 / mse).item()
        ssim = structural_similarity(x.cpu().numpy(), orig.cpu().numpy(), win_size=21, channel_axis=0, data_range=1.0)
        # Sparse reward (SSIM)
        if done and ssim > self.final_threshold:
            reward += 1

        return reward, ssim, psnr
    
    def render(self, mode='human', close=False):
        # This could visualize the current state if necessary
        pass
