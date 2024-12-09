from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder, SubprocVecEnv
from stable_baselines3 import A2C, DQN, PPO, SAC
from gymnasium import spaces
import torch as th
import torch.nn as nn
import wandb
from wandb.integration.sb3 import WandbCallback
import warnings
import gymnasium as gym
from gymnasium.envs.registration import register
import torch.nn.functional as F
# from func import MD_SAC
import os
from main import parse_args_and_config
from guided_diffusion.my_diffusion import Diffusion as my_diffusion
from train import CustomCNN
from tqdm import tqdm

th.set_printoptions(sci_mode=False)

warnings.filterwarnings("ignore")
register(
    id='final-eval',
    entry_point='envs:EvalDiffusionEnv',
)

def make_env(my_config):
    def _init():
        config = {
            "target_steps": my_config["target_steps"],
            "max_steps": my_config["max_steps"],
            "threshold": my_config["threshold"],
            "DM": my_config["DM"],
        }
        return gym.make('final-eval', **config)
    return _init

    
def evaluation(env, model, eval_num=100):
    avg_ssim = 0
    avg_psnr = 0
    ### Run eval_num times rollouts,
    for _ in tqdm(range(eval_num)):
        done = False
        # Set seed and reset env using Gymnasium API
        obs = env.reset()

        while not done:
            # Interact with env using Gymnasium API
            action, _state = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
        avg_ssim += info[0]['ssim']
        avg_psnr += info[0]['psnr']
    avg_ssim /= eval_num
    avg_psnr /= eval_num

    return avg_ssim, avg_psnr

def main():
    # Initialze DDNM
    args, config = parse_args_and_config()
    runner = my_diffusion(args, config)

    policy_kwargs = dict(
        features_extractor_class=CustomCNN,
        features_extractor_kwargs=dict(features_dim=256),
    )
    my_config = {
        "algorithm": A2C,
        "target_steps": args.target_steps,
        "threshold": 0.9,
        "policy_network": "MultiInputPolicy",
        "policy_kwargs": policy_kwargs,
        "max_steps": 100,
        "num_eval_envs": 1,
        "eval_num": len(runner.test_dataset),
    }
    my_config['save_path'] = f'model/sample_model_A2C_{my_config["target_steps"]}/{args.eval_model_idx}'
    config = {
            "target_steps": my_config["target_steps"],
            "max_steps": my_config["max_steps"],
            "threshold": my_config["threshold"],
            "DM": runner,
        }

    ### Load model with SB3
    model = A2C.load(my_config['save_path'])
    print("Loaded model from: ", my_config['save_path'])
    env = DummyVecEnv([make_env(config) for _ in range(my_config['num_eval_envs'])])
    
    avg_ssim, avg_psnr = evaluation(env, model, my_config['eval_num'])

    print(f"Counts: (Total of {my_config['eval_num']} rollouts)")
    print("Total Average PSNR: %.2f" % avg_psnr)
    print("Total Average SSIM: %.3f" % avg_ssim)


if __name__ == '__main__':
    main()
