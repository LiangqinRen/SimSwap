import inspect

import simswap_defense
import utils


def main():
    args = utils.get_argparser()
    logger = utils.get_file_and_console_logger(args)

    utils.check_cuda_availability(logger)
    utils.fix_random_seed(args, logger)
    utils.show_parameters(args, logger)
    timer = utils.Timer(inspect.currentframe().f_code.co_name, logger)

    defense = simswap_defense.SimSwapDefense(args, logger)
    defense_functions = {
        "split": defense.split_dataset,
        "swap": defense.swap,
        "metric": defense.calculate_efficiency_threshold,
        "pgd_src_single": defense.pgd_source_single,
        "pgd_src_multi": defense.pgd_source_multiple,
        "pgd_tgt_single": defense.pgd_target_single,
        "pgd_tgt_multi": defense.pgd_target_multiple,
        "gan_src": defense.GAN_SRC,
        "gan_tgt": defense.GAN_TGT,
        "gan_src_test": defense.GAN_SRC_test,
        "gan_tgt_test": defense.GAN_TGT_test,
        "gan_src_robust": defense.GAN_SRC_robust,
        "gan_tgt_robust": defense.GAN_TGT_robust,
        "test": defense._get_average_identity,
    }

    if args.method in defense_functions:
        defense_functions[args.method]()
    else:
        raise NotImplementedError


if __name__ == "__main__":
    main()
