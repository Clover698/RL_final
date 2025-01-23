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
import os
from main import parse_args_and_config
from guided_diffusion.my_diffusion import Diffusion as my_diffusion
from train import CustomCNN
from tqdm import tqdm

th.set_printoptions(sci_mode=False)

warnings.filterwarnings("ignore")
register(
    id='final-ours',
    entry_point='envs:EvalDiffusionEnv',
)
register(
    id='final-baseline',
    entry_point='envs:BaselineEvalDiffusionEnv',
)

def make_env(my_config):
    def _init():
        config = {
            "target_steps": my_config["target_steps"],
            "max_steps": my_config["max_steps"],
            "threshold": my_config["threshold"],
            "DM": my_config["DM"],
            # "agent1": my_config["agent1"],
        }
        if my_config["model_mode"] == "baseline":
            print('Baseline training mode ...')
            return gym.make('final-baseline', **config)
        else:
            print('2-agent training mode ...')
            config["agent1"] = my_config["agent1"]
            return gym.make('final-ours', **config)
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
        print(info[0]['time_step_sequence'], info[0]['ssim'], info[0]['psnr'])
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
    my_config['save_path'] = f'model/{args.eval_model_name}/best'

    ### Load agent of subtask1 with SB3
    agent1 = A2C.load(my_config['save_path'])
    print("Loaded model from: ", my_config['save_path'])

    config = {
            "target_steps": my_config["target_steps"],
            "max_steps": my_config["max_steps"],
            "threshold": my_config["threshold"],
            "DM": runner,
            # "agent1": agent1,
            "model_mode": "baseline" if args.baseline else "2_agents",
        }
    # Load agent of subtask 2
    if args.baseline == False:
        if args.subtask1 == False:
            agent2 = A2C.load(my_config['save_path'] + '_2')
        config["agent1"] = None if args.subtask1 else agent1

    env = DummyVecEnv([make_env(config) for _ in range(my_config['num_eval_envs'])])

    if args.baseline or args.subtask1:
        avg_ssim, avg_psnr = evaluation(env, agent1, my_config['eval_num'])
    else:
        avg_ssim, avg_psnr = evaluation(env, agent2, my_config['eval_num'])

    print(f"Counts: (Total of {my_config['eval_num']} rollouts)")
    print("Total Average PSNR: %.2f" % avg_psnr)
    print("Total Average SSIM: %.3f" % avg_ssim)


if __name__ == '__main__':
    main()
