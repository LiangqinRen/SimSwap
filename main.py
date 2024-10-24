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
        # "split": defense.split_dataset,
        # "swap": defense.swap,
        # "metric": defense.calculate_efficiency_threshold,
        # "pgd_source_single": defense.pgd_source_single,
        # "pgd_source_multi": defense.pgd_source_multiple,
        # "pgd_source_metric": defense.pgd_source_metric,
        # "pgd_source_distance": defense.pgd_source_distance,
        # "pgd_target_single": defense.pgd_target_single,
        # "pgd_target_multi": defense.pgd_target_multiple,
        # "pgd_target_metric": defense.pgd_target_metric,
        # "gan_source": defense.gan_source,
        # "gan_target": defense.gan_target,
        #
        # "gan_target_metric": defense.gan_target_metric,
        #
        #
        # "pgd_source_robustness_metric": defense.pgd_source_robustness_metric,
        # "pgd_target_robustness_sample": defense.pgd_target_robustness_sample,
        # "pgd_target_robustness_metric": defense.pgd_target_robustness_metric,
        #
        "pgd_source_sample": defense.pgd_source_sample,
        "pgd_target_sample": defense.pgd_target_sample,
        "pgd_source_metric": defense.pgd_source_metric,
        "pgd_target_metric": defense.pgd_target_metric,
        "pgd_source_robustness_sample": defense.pgd_source_robustness_sample,
        "pgd_target_robustness_sample": defense.pgd_target_robustness_sample,
        "pgd_source_robustness_metric": defense.pgd_source_robustness_metric,
        "pgd_target_robustness_metric": defense.pgd_target_robustness_metric,
        "gan_source_sample": defense.gan_source_sample,
        "gan_source_metric": defense.gan_source_metric,
        "gan_source_robustness_sample": defense.gan_source_robustness_sample,
        "gan_target_robustness_sample": defense.gan_target_robustness_sample,
        "gan_source_robustness_metric": defense.gan_source_robustness_metric,
        "gan_target_robustness_metric": defense.gan_target_robustness_metric,
    }

    if args.method in defense_functions:
        defense_functions[args.method]()
    else:
        raise NotImplementedError


if __name__ == "__main__":
    main()
