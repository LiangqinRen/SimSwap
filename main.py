import simswap_defense
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
        "pgd_both_sample": defense.pgd_both_sample,
        "pgd_both_metric": defense.pgd_both_metric,
        "pgd_both_robustness_sample": defense.pgd_both_robustness_sample,
        "pgd_both_robustness_metric": defense.pgd_both_robustness_metric,
        "pgd_robustness_forensics_sample": defense.pgd_robustness_forensics_sample,
        "pgd_robustness_forensics_metric": defense.pgd_robustness_forensics_metric,
        "gan_both_train": defense.gan_both_train,
        "gan_both_sample": defense.gan_both_sample,
        "gan_both_metric": defense.gan_both_metric,
        "gan_both_robustness_sample": defense.gan_both_robustness_sample,
        "gan_both_robustness_metric": defense.gan_both_robustness_metric,
        "anchor_difference": defense.check_img_anchor_difference,
    }

    if args.method in defense_functions:
        defense_functions[args.method]()
    elif args.method == "robustness":
        from robustness import main

        main(args, logger)
    else:
        raise NotImplementedError


if __name__ == "__main__":
    main()
