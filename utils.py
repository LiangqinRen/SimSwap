import cv2
import torch
import logging
import time
import argparse
import pathlib
import random
import os

import numpy as np


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
        1: logging.DEBUG,
        2: logging.INFO,
        3: logging.WARNING,
        4: logging.ERROR,
        5: logging.FATAL,
    }

    formatter = logging.Formatter(
        fmt="[%(asctime)s.%(msecs)03d][%(filename)10s:%(lineno)4d][%(levelname)5s]|%(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    current_time = time.strftime("%Y:%m:%d-%H:%M:%S", time.localtime(time.time()))
    args.log_dir = f"{log_folder}/{current_time}"
    pathlib.Path(f"{args.log_dir}").mkdir(parents=True, exist_ok=True)
    pathlib.Path(f"{args.log_dir}/image").mkdir(parents=True, exist_ok=True)

    log_path = f"{log_folder}/{current_time}/{current_time}.log"
    handler_to_file = logging.FileHandler(log_path, mode="w")
    handler_to_file.setFormatter(formatter)
    handler_to_console = logging.StreamHandler()
    handler_to_console.setFormatter(formatter)

    logger = logging.getLogger()
    logger.setLevel(log_levels[log_level])
    logger.addHandler(handler_to_file)
    logger.addHandler(handler_to_console)

    with open(os.path.join(args.log_dir, "notes.txt"), "w") as f:
        pass

    return logger


def get_argparser():
    parser = argparse.ArgumentParser(description="Thwart DeepFake!")
    parser.add_argument("--log_level", type=int, default=2)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--method", type=str)
    parser.add_argument("--data_dir", type=str, default="crop_224")
    parser.add_argument("--random_seed", type=int, default=0)

    parser.add_argument("--metric_people_image", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--testset_percentage", type=int, default=10)
    parser.add_argument(
        "--effectiveness_threshold", type=float, default=1.0492130517959595
    )

    parser.add_argument("--pgd_epsilon", type=float, default=1e-2)
    parser.add_argument("--pgd_limit", type=float, default=5e-2)

    parser.add_argument("--gan_generator_lr", type=float, default=5e-4)
    parser.add_argument("--gan_test_models", type=str)

    args = parser.parse_args()

    return args


def show_parameters(args, logger) -> None:
    content = "Parameter configuration:\n"

    for arg in vars(args).keys():
        content += f"\t{arg}: {getattr(args, arg)}\n"

    logger.info(content)


def fix_random_seed(args, logger) -> None:
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)

    logger.info(f"Fix random, numpy and torch random seed to {args.random_seed}")
