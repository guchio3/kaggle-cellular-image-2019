import os
import time
from logging import getLogger

from tools.utils.args import parse_args
from tools.utils.configs import load_configs
from tools.utils.logs import logInit, sel_log, send_line_notification


CONFIG_DIR = './configs/exp_configs/'

if __name__ == '__main__':
    t0 = time.time()
    script_name = os.path.basename(__file__).split('.')[0]
    args = parse_args(None)
    exp_id = args.exp_id
    log_file = f'{script_name}_{exp_id}.log'

    logger = getLogger(__name__)
    logger = logInit(logger, './mnt/logs/', log_file)
    sel_log(f'args: {sorted(vars(args).items())}', logger)
    configs = load_configs(CONFIG_DIR + exp_id + '.yml', logger)

    if configs['runner'] == 'r001':
        from tools.runners.r001_basic_runner import Runner
    runner = Runner(configs, args, logger)
    runner.train_model()

    prec_time = time.time() - t0
#    send_line_notification(f'Finished: {script_name} '
#                           f'using CONFIG: {exp_id} '
#                           f'w/ MAPE {mape_mean:.5f}+-{mape_std:.5f} '
#                           f'in {prec_time:.1f} s !')
