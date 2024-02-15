import torch
import logging
import time
import argparse
import enum
import pathlib


class ProtectMode(enum.Enum):
    void = 0  # non protection, test the original deepfake function
    GAN = 1
    PGD = 2


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
    log_folder = "./log"
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
    pathlib.Path(f"{log_folder}/{current_time}").mkdir(parents=True, exist_ok=True)
    args.ID = current_time

    log_path = f"{log_folder}/{current_time}/{current_time}.log"
    handler_to_file = logging.FileHandler(log_path, mode="w")
    handler_to_file.setFormatter(formatter)
    handler_to_console = logging.StreamHandler()
    handler_to_console.setFormatter(formatter)

    logger = logging.getLogger()
    logger.setLevel(log_levels[log_level])
    logger.addHandler(handler_to_file)
    logger.addHandler(handler_to_console)

    return logger


def get_argparser():
    parser = argparse.ArgumentParser(description="Thwart DeepFake!")
    parser.add_argument(
        "--log_level",
        type=int,
        default=2,
        help="log level",
    )
    parser.add_argument(
        "--ID",
        type=str,
        default="",
        help="identify the experiments, the default ID is the timestamp",
    )
    parser.add_argument(
        "--project",
        type=str,
        default="",
        help="the project to protect, such as simswap",
    )
    parser.add_argument(
        "--protect_mode",
        type=int,
        help="the protect mode for the project",
    )
    parser.add_argument(
        "--use_disc",
        action="store_true",
        help="use the discriminator",
    )
    parser.add_argument(
        "--save_interval",
        type=int,
        default=100,
        help="intermidiate results save interval",
    )
    parser.add_argument(
        "--epoch",
        type=int,
        default=30000,
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
    )

    args = parser.parse_args()

    return args


def show_parameters(args, logger) -> None:
    content = "Parameter configuration:\n"

    for arg in vars(args).keys():
        content += f"\t{arg}: {getattr(args, arg)}\n"

    logger.info(content)