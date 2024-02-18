import sys

# sys.path.append("../third_party/Faceswap-Deepfake-Pytorch")
sys.path.append("../")

import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"

import utils

# import faceswap_defense
import simswap_defense

import time


if __name__ == "__main__":
    start_time = time.time()
    args = utils.get_argparser()
    logger = utils.get_file_and_console_logger(args)

    utils.check_cuda_availability(logger)
    utils.show_parameters(args, logger)

    defense_classes = {
        # "faceswap": faceswap_defense.FaceSwapDefense,
        "simswap": simswap_defense.SimSwapDefense,
    }

    defense = defense_classes[args.project](logger, args)
    if utils.ProtectMode(args.protect_mode) == utils.ProtectMode.void:
        defense.void(args)
    elif utils.ProtectMode(args.protect_mode) == utils.ProtectMode.PGD:
        defense.PGD(args)
    elif utils.ProtectMode(args.protect_mode) == utils.ProtectMode.GAN:
        defense.GAN_clip(args)
    else:
        raise Exception(f"Unknown protect mode [{args.protect_mode}]!")

    logger.info(f"DeepFake protection costs {time.time() - start_time:.3f} seconds")
