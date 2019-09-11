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
    elif configs['runner'] == 'r002':
        from tools.runners.r002_warmup_separate import Runner
    elif configs['runner'] == 'r003':
        from tools.runners.r003_metric_learning import Runner
    runner = Runner(configs, args, logger)
    if not args.prediction:
        if len(args.cell_types) != 1:
            raise Exception('you can use just one cell type for train')
        cell_type = args.cell_types[0]
        runner.train_model(cell_type)
    else:
        sub_filename = None
        for i, cell_type in enumerate(args.cell_types):
            sub_filename = runner.make_submission_file(cell_type, sub_filename)

    send_line_notification(f'Finished cell_type {args.cell_types}')
