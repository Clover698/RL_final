import torch
import numpy as np
import shutil
import sys
import math
import logging
import yaml
import argparse
import os


def parse_args_and_config():
    parser = argparse.ArgumentParser(description=globals()["__doc__"])

    parser.add_argument(
        "--config", type=str, required=True, help="Path to the config file"
    )
    parser.add_argument("--seed", type=int, default=1234, help="Random seed")
    parser.add_argument(
        "--exp", type=str, default="ddrm/exp", help="Path for saving running related data."
    )
    parser.add_argument(
        "--doc",
        type=str,
        required=True,
        help="A string for documentation purpose. "
        "Will be the name of the log folder.",
    )
    parser.add_argument(
        "--comment", type=str, default="", help="A string for experiment comment"
    )
    parser.add_argument(
        "--verbose",
        type=str,
        default="info",
        help="Verbose level: info | debug | warning | critical",
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Whether to produce samples from the model",
    )
    parser.add_argument(
        "-i",
        "--image_folder",
        type=str,
        default="images",
        help="The folder name of samples",
    )
    parser.add_argument(
        "--ni",
        action="store_true",
        help="No interaction. Suitable for Slurm Job launcher",
    )
    parser.add_argument(
        "--target_steps", type=int, default=10, help="The target step of the model"
    )
    parser.add_argument(
        "--timesteps", type=int, default=1000, help="number of steps involved"
    )
    parser.add_argument("--deg", type=str, required=True, help="Degradation")
    parser.add_argument("--sigma_0", type=float, required=True, help="Sigma_0")
    parser.add_argument("--eta", type=float, default=0.85, help="Eta")
    parser.add_argument("--etaB", type=float, default=1, help="Eta_b (before)")
    parser.add_argument("--subset_start", type=int, default=-1)
    parser.add_argument("--subset_end", type=int, default=-1)
    parser.add_argument("--input_root", type=str, default="/disk_195a/qiannnhui", help="The root folder of input images")
    parser.add_argument("--second_stage", action="store_true", help="Whether to run the second stage")
    parser.add_argument("--RL_algorithm", type=str, default="SAC", help="The RL algorithm to use")
    parser.add_argument("--eval_model_name", type=str, default="SR_baseline_A2C_10", help="eval_model_name")

    args = parser.parse_args()
    args.log_path = os.path.join(args.exp, "logs", args.doc)

    # parse config file
    with open(os.path.join("ddrm/configs", args.config), "r") as f:
        config = yaml.safe_load(f)
    new_config = dict2namespace(config)

    # tb_path = os.path.join(args.exp, "tensorboard", args.doc)

    level = getattr(logging, args.verbose.upper(), None)
    if not isinstance(level, int):
        raise ValueError("level {} not supported".format(args.verbose))

    handler1 = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(levelname)s - %(filename)s - %(asctime)s - %(message)s"
    )
    handler1.setFormatter(formatter)
    logger = logging.getLogger()
    logger.addHandler(handler1)
    logger.setLevel(level)

    os.makedirs(os.path.join(args.exp, "image_samples"), exist_ok=True)
    args.image_folder = os.path.join(args.exp, "image_samples", args.image_folder)
    if not os.path.exists(args.image_folder):
        os.makedirs(args.image_folder)
    else:
        overwrite = False
        if args.ni:
            overwrite = True
        else:
            response = input(
                f"Image folder {args.image_folder} already exists. Overwrite? (Y/N)"
            )
            if response.upper() == "Y":
                overwrite = True

        if overwrite:
            shutil.rmtree(args.image_folder)
            os.makedirs(args.image_folder)
        else:
            print("Output image folder exists. Program halted.")
            sys.exit(0)

    # add device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    logging.info("Using device: {}".format(device))
    new_config.device = device

    # set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    torch.backends.cudnn.benchmark = True

    return args, new_config


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace
