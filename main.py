import simswap_defense
from miscellaneous import Worker
import utils

import inspect


def main():
    args = utils.get_argparser()
    logger = utils.get_file_and_console_logger(args)

    utils.check_cuda_availability(logger)
    utils.fix_random_seed(args, logger)
    utils.show_parameters(args, logger)
    timer = utils.Timer(inspect.currentframe().f_code.co_name, logger)

    defense = simswap_defense.SimSwapDefense(args, logger)
    defense_functions = {
        "pgd_source_distance": defense.pgd_source_distance,
        "pgd_both_sample": defense.pgd_both_sample,
        "pgd_both_metric": defense.pgd_both_metric,
        "pgd_source_robustness_sample": defense.pgd_source_robustness_sample,
        "pgd_target_robustness_sample": defense.pgd_target_robustness_sample,
        "pgd_source_robustness_metric": defense.pgd_source_robustness_metric,
        "pgd_target_robustness_metric": defense.pgd_target_robustness_metric,
        "gan_source_sample": defense.gan_source_sample,
        "gan_target_sample": defense.gan_target_sample,
        "gan_source_metric": defense.gan_source_metric,
        "gan_target_metric": defense.gan_target_metric,
        "gan_source_robustness_sample": defense.gan_source_robustness_sample,
        "gan_target_robustness_sample": defense.gan_target_robustness_sample,
        "gan_source_robustness_metric": defense.gan_source_robustness_metric,
        "gan_target_robustness_metric": defense.gan_target_robustness_metric,
    }

    if args.method in defense_functions:
        defense_functions[args.method]()
    elif args.method == "worker":
        from miscellaneous import main

        main(args, logger)
    else:
        raise NotImplementedError


if __name__ == "__main__":
    main()
