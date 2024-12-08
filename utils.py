import torch
import logging
import time
import argparse
import pathlib
import random
import os
import json

import numpy as np
from os.path import join


class Timer:
    def __init__(self, function_name, logger):
        self.function_name = function_name
        self.begin_time = time.time()
        self.logger = logger

    def __del__(self):
        self.logger.info(
            f"{self.function_name} costs {time.time() - self.begin_time:.3f} seconds"
        )


def check_cuda_availability(logger):
    if torch.cuda.is_available():
        logger.info(f"Use GPU {torch.cuda.get_device_name()}")
    else:
        raise SystemExit("CUDA is not available!")


def get_file_and_console_logger(args):
    log_folder = "log"
    log_level = args.log_level
    log_levels = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
        "fatal": logging.FATAL,
    }

    formatter = logging.Formatter(
        fmt="[%(asctime)s.%(msecs)03d][%(filename)10s:%(lineno)4d][%(levelname)5s]|%(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logger = logging.getLogger()
    current_time = time.strftime("%Y:%m:%d-%H:%M:%S", time.localtime(time.time()))
    args.log_dir = join(log_folder, current_time if not args.console_only else "null")

    pathlib.Path(f"{args.log_dir}").mkdir(parents=True, exist_ok=True)
    pathlib.Path(f"{args.log_dir}/image").mkdir(parents=True, exist_ok=True)

    log_path = join(args.log_dir, f"{current_time}.log")
    handler_to_file = logging.FileHandler(log_path, mode="w")
    handler_to_file.setFormatter(formatter)
    logger.addHandler(handler_to_file)

    with open(os.path.join(args.log_dir, "notes.txt"), "w") as f:
        pass

    handler_to_console = logging.StreamHandler()
    handler_to_console.setFormatter(formatter)

    logger.setLevel(log_levels[log_level])
    logger.addHandler(handler_to_console)

    try:
        with open(os.path.join(args.data_dir, "facepp_keys.json")) as f:
            data = json.load(f)
            args.facepp_api_key = data["api_key"]
            args.facepp_api_secret = data["api_secret"]
    except Exception as e:
        raise SystemExit("Can't find facepp_keys.json!")

    try:
        with open(os.path.join(args.data_dir, "aws_keys.json")) as f:
            data = json.load(f)
            args.aws_api_key = data["api_key"]
            args.aws_api_secret = data["api_secret"]
            args.aws_api_region = data["api_region"]
    except Exception as e:
        raise SystemExit("Can't find aws_keys.json!")

    return logger


def get_argparser():
    parser = argparse.ArgumentParser(description="Defense Face Swap")
    parser.add_argument("--log_level", type=str, default="info")
    parser.add_argument(
        "--console_only",
        action="store_true",
        help="Enable console-only logging mode. Logs will not be written to a file",
    )
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--method", type=str)
    parser.add_argument("--data_dir", type=str, default="dataset")
    parser.add_argument("--anchor_dir", type=str, default="real")
    parser.add_argument("--anchor_index", type=int, default=0)
    parser.add_argument("--random_seed", type=int, default=0)

    parser.add_argument("--metric_people_image", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--testset_percentage", type=int, default=10)

    parser.add_argument("--pgd_epsilon", type=float, default=1e-2)
    parser.add_argument("--pgd_limit", type=float, default=5e-2)

    parser.add_argument("--gan_generator_lr", type=float, default=5e-4)
    parser.add_argument("--gan_test_models", type=str)

    args = parser.parse_args()

    return args


def show_parameters(args, logger) -> None:
    sensitive_word_list = [
        "facepp_api_key",
        "facepp_api_secret",
        "aws_api_key",
        "aws_api_secret",
    ]

    content = "Parameter configuration:\n"
    for arg in vars(args).keys():
        if arg in sensitive_word_list:
            if isinstance(getattr(args, arg), list):
                content += f"\t{arg}: [HIDDEN](count: {len(getattr(args, arg))})\n"
            else:
                content += f"\t{arg}: [HIDDEN]\n"
        else:
            content += f"\t{arg}: {getattr(args, arg)}\n"

    logger.info(content)


def fix_random_seed(args, logger) -> None:
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)

    logger.info(f"Fix random, numpy and torch random seed to {args.random_seed}")
