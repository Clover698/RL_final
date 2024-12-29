import gymnasium as gym
from gymnasium.envs.registration import register
from stable_baselines3 import A2C, PPO, DQN, SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder, SubprocVecEnv
from train import CustomCNN
import numpy as np
from collections import Counter
from func import MD_SAC
from tqdm import tqdm
import argparse

from ddrm.runners.diffusion import Diffusion
from arguments import parse_args_and_config

register(
    id='final-eval',
    entry_point='envs:EvalDiffusionEnv'
)

# def arg_parser():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--run_id", type=str, default="SAC_v1")
#     parser.add_argument("--save_path", type=str, default="model/sample_model/262")
#     parser.add_argument("--DM_model", type=str, default="model/ddpm_ema_cifar10")
#     parser.add_argument("--target_steps", type=int, default=100)
#     parser.add_argument("--img_save_path", type=str, default="run/sample/SAC_DDIM_100")
#     parser.add_argument("--action_range", type=float, default=1.0, help="Range of the action")
#     return parser.parse_args()

def make_env(my_config):
    def _init():
        config = {
            "runner": my_config["runner"],
            "target_steps": my_config["target_steps"],
            "max_steps": my_config["max_steps"],
            "agent1": my_config["agent1"],
        }
        return gym.make("final-eval", **config)

    return _init
    
def evaluation(env, model, eval_num=100):
    avg_ssim = 0
    avg_psnr = 0
    ### Run eval_num times rollouts,
    for i in tqdm(range(eval_num)):
        done = False
        # Set seed and reset env using Gymnasium API
        obs = env.reset()

        while not done:
            # Interact with env using Gymnasium API
            action, _state = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
        
        time_step_sequence = info[0]["time_step_sequence"]
        avg_ssim += info[0]['ssim']
        avg_psnr += info[0]['psnr']
    avg_ssim /= eval_num
    avg_psnr /= eval_num

    return avg_ssim, avg_psnr, time_step_sequence


def main():
    # Initialze DDNM

    args, config = parse_args_and_config()
    with open("./output.txt", "a") as file:
        file.write("==========================\n")
        file.write(f'{args.eval_model_name}\n')
    runner = Diffusion(args, config)
    runner.sample()

    policy_kwargs = dict(
        features_extractor_class=CustomCNN,
        features_extractor_kwargs=dict(features_dim=256),
    )
    my_config = {
        "algorithm": A2C,
        "target_steps": args.target_steps,
        "policy_network": "MultiInputPolicy",
        "policy_kwargs": policy_kwargs,
        "max_steps": 100,
        "num_eval_envs": 1,
        "runner": runner,
        "eval_num": len(runner.test_dataset),
    }
    my_config['save_path'] = f'model/{args.eval_model_name}/best'

    ### Load model with SB3
    agent1 = my_config['algorithm'].load(my_config['save_path'])
    agent2 = my_config['algorithm'].load(my_config['save_path'] + '_2')
    print("Loaded model from: ", my_config['save_path'])

    config = {
            "runner": my_config["runner"],
            "target_steps": my_config["target_steps"],
            "max_steps": my_config["max_steps"],
            "agent1": agent1,
        }

    env = DummyVecEnv([make_env(config) for _ in range(my_config['num_eval_envs'])])
    
    avg_ssim, avg_psnr, time_step_sequence = evaluation(env, agent2, my_config['eval_num'])

    print(f"Counts: (Total of {my_config['eval_num']} rollouts)")
    print("Total Average PSNR: %.2f" % avg_psnr)
    print("Total Average SSIM: %.3f" % avg_ssim)
    with open("./output.txt", 'a') as file:
        file.write(f"Counts: (Total of {my_config['eval_num']} rollouts)\n")
        file.write(f"time step sequence = {time_step_sequence}\n")
        file.write("Total Average SSIM: %.3f" % avg_ssim)
        file.write("\n")
        file.write("Total Average PSNR: %.3f" % avg_psnr)
        file.write("\n")

if __name__ == "__main__":
    main()