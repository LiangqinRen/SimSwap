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
        "swap": defense.swap,
        "pgd_src_single": defense.pgd_src_single,
        "pgd_src_multi": defense.pgd_src_multi,
    }

    if args.method in defense_functions:
        defense_functions[args.method]()
    else:
        raise NotImplementedError


if __name__ == "__main__":
    main()
