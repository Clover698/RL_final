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
import gc

class DiffusionEnv(gym.Env):
    def __init__(self, target_steps=10, max_steps=100, DM=None, agent1=None):
        super(DiffusionEnv, self).__init__()
        self.DM = DM
        self.agent1 = agent1 # RL model from subtask 1
        self.target_steps = target_steps
        self.uniform_steps = [i for i in range(0, 999, 1000//target_steps)][::-1]
        # adjust: False -> First subtask, True -> Second subtask
        self.adjust = True if agent1 is not None else False
        
        self.sample_size = 256
        # Maximum number of steps  (Baseline)
        self.max_steps = max_steps 
        # Count the number of steps
        self.current_step_num = 0 
        if self.adjust:
            self.action_space = gym.spaces.Box(low=-5, high=5)
        else:
            self.action_space = spaces.Discrete(20)
        # Define the action and observation space
        self.observation_space = Dict({
            "image": Box(low=-1, high=1, shape=(3, self.sample_size, self.sample_size), dtype=np.float32),
            "value": Box(low=np.array([0]), high=np.array([999]), dtype=np.uint16)
        })
        # Initialize the random seed
        self.seed(232)
        self.reset()
        # print("Training data size:", len(self.DM.dataset))
        
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
        self.data_idx = random.randint(0, len(self.DM.dataset)-1)
        self.x_orig, self.classes = self.DM.dataset[self.data_idx]
        self.x, self.y, self.Apy, self.x_orig, self.A_inv_y = self.DM.preprocess(self.x_orig, self.data_idx)
        ddim_x = self.x.clone()
        ddim_x0_t = self.A_inv_y.clone()
        self.x0_t = self.A_inv_y.clone()
        with torch.no_grad():
            for i in range(self.target_steps):
                ddim_t = torch.tensor(self.uniform_steps[i])
                if i != 0:
                    ddim_x = self.DM.get_noisy_x(ddim_t, ddim_x0_t, self.ddim_et)
                # else:
                #     self.ddim_x = self.DM.get_noisy_x(ddim_t, self.ddim_x0_t, initial=True)
                ddim_x0_t, _,  self.ddim_et = self.DM.single_step_ddnm(ddim_x, self.y, ddim_t, self.classes)
        orig = inverse_data_transform(self.DM.config, self.x_orig[0]).to(self.DM.device)
        ddim_x = inverse_data_transform(self.DM.config, ddim_x0_t[0]).to(self.DM.device)
        ddim_mse = torch.mean((ddim_x - orig) ** 2)
        self.ddim_psnr = 10 * torch.log10(1 / ddim_mse).item()
        self.ddim_ssim = structural_similarity(ddim_x.cpu().numpy(), orig.cpu().numpy(), win_size=21, channel_axis=0, data_range=1.0)
        del ddim_x, ddim_x0_t, ddim_mse, orig
        gc.collect()

        observation = {
            "image": self.x0_t[0].cpu(),  
            "value": np.array([999])
        }
        if self.adjust: # Second subtask
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
                self.current_step_num += 1

        torch.cuda.empty_cache()  # Clear GPU cache
        # images = (self.GT_image / 2 + 0.5).clamp(0, 1)
        # images = images.cpu().permute(0, 2, 3, 1).numpy()[0]
        # images = Image.fromarray((images * 255).round().astype("uint8"))
        # filename = os.path.join('img', f"GT_{self.current_step_num}.png")
        # images.save(filename)
        return observation, {}
    
    def step(self, action):
        truncate = True if self.current_step_num >= self.max_steps else False
        # Denoise current image at time t
        with torch.no_grad():
            ### RL step
            if self.adjust == False: # First subtask
                start_t = 50 * (1+action) - 1
                t = torch.tensor(int(max(0, min(start_t, 999))))
                self.interval = int(t / (self.target_steps - 1)) 
                self.x = self.DM.get_noisy_x(t, self.x0_t, initial=True)
                self.action_sequence.append(action.item())
            else: # Second subtask
                t = self.previous_t - self.interval - self.interval * float(action - 100.0) / 200.0
                t = torch.tensor(int(max(0, min(t, 999))))
                self.interval = int(t / (self.target_steps - self.current_step_num - 1)) if (self.target_steps - self.current_step_num - 1) != 0 else self.interval
                self.x = self.DM.get_noisy_x(t, self.x0_t, self.et)
                self.action_sequence.append(action.item())
            self.previous_t = t
            self.x0_t, _,  self.et = self.DM.single_step_ddnm(self.x, self.y, t, self.classes)
            self.time_step_sequence.append(t.item())

            self.uniform_x0_t = self.x0_t.clone()
            self.uniform_et = self.et.clone()
            for i in range(self.target_steps - self.current_step_num - 1): # Run remaining steps via uniform policy
                uniform_t = torch.tensor(int(t - self.interval - self.interval * i))
                uniform_t = torch.tensor(max(0, min(uniform_t, 999)))
                self.uniform_x = self.DM.get_noisy_x(uniform_t, self.uniform_x0_t, self.uniform_et)
                self.uniform_x0_t, _,  self.uniform_et = self.DM.single_step_ddnm(self.uniform_x, self.y, uniform_t, self.classes)

        # Finish the episode if denoising is done
        done = (self.current_step_num == self.target_steps - 1) or not self.adjust
        # Calculate reward
        reward, ssim, psnr, ddim_ssim, ddim_psnr = self.calculate_reward(done)
        # if done:
        #     self.DM.postprocess(self.x0_t, self.x_orig, self.data_idx)
        info = {
            'ddim_t': self.uniform_steps[self.current_step_num],
            't': t,
            'reward': reward,
            'ssim': ssim,
            'psnr': psnr,
            'ddim_ssim': ddim_ssim,
            'ddim_psnr': ddim_psnr,
            'time_step_sequence': self.time_step_sequence,
            'action_sequence': self.action_sequence,
        }
        # print('info:', info)
        observation = {
            "image":  self.x0_t[0].cpu(),  
            "value": np.array([t])
        }
        # Increase number of steps
        self.current_step_num += 1
        torch.cuda.empty_cache()  # Clear GPU cache
        return observation, reward, done, truncate, info

    def calculate_reward(self, done):
        reward = 0
        orig = inverse_data_transform(self.DM.config, self.x_orig[0]).to(self.DM.device)
        if done and self.adjust:
            x = inverse_data_transform(self.DM.config, self.x0_t[0]).to(self.DM.device)
        else:
            x = inverse_data_transform(self.DM.config, self.uniform_x0_t[0]).to(self.DM.device)
        mse = torch.mean((x - orig) ** 2)
        psnr = 10 * torch.log10(1 / mse).item()
        ssim = structural_similarity(x.cpu().numpy(), orig.cpu().numpy(), win_size=21, channel_axis=0, data_range=1.0)
        
        # Intermediate reward (Percentage of temporary improvement)
        if not done and psnr > self.ddim_psnr and ssim > self.ddim_ssim:
            reward += 0.5/self.target_steps*psnr/self.ddim_psnr 
            reward += 0.5/self.target_steps*ssim/self.ddim_ssim
        
        # Sparse reward (Percentage of final improvement)
        if done and psnr > self.ddim_psnr and ssim > self.ddim_ssim:
            reward += 0.5*psnr/self.ddim_psnr
            reward += 0.5*ssim/self.ddim_ssim


        return reward, ssim, psnr, self.ddim_ssim, self.ddim_psnr
    
    def render(self, mode='human', close=False):
        # This could visualize the current state if necessary
        pass

    def set_adjust(self, adjust):
        self.adjust = adjust
        print(f"Set adjust to {adjust}")
