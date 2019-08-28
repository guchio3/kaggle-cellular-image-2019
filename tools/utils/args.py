import argparse

from .logs import sel_log


def parse_test_args(logger=None):
    '''
    Policy
    ------------
    * experiment id must be required

    '''
    parser = argparse.ArgumentParser(
        prog='XXX.py',
        usage='ex) python -e e001 -d -m "e001, basic experiment"',
        description='short explanation of args',
        add_help=True,
    )
    parser.add_argument('-e', '--exp_config',
                        help='experiment setting',
                        type=str,
                        required=True)
    parser.add_argument('-t', '--trained_model',
                        help='messages about the process',
                        type=str,
                        required=True)
    parser.add_argument('-m', '--message',
                        help='messages about the process',
                        type=str,
                        default='')

    args = parser.parse_args()
    sel_log(f'args: {sorted(vars(args).items())}', logger)
    return args


def parse_train_args(logger=None):
    '''
    Policy
    ------------
    * experiment id must be required

    '''
    parser = argparse.ArgumentParser(
        prog='XXX.py',
        usage='ex) python -e e001 -d -m "e001, basic experiment"',
        description='short explanation of args',
        add_help=True,
    )
    parser.add_argument('-e', '--exp_config',
                        help='experiment setting',
                        type=str,
                        required=True)
#    parser.add_argument('-t', '--test',
#                        help='set when you run test',
#                        action='store_true',
#                        default=False)
    parser.add_argument('-d', '--debug',
                        help='whether or not to use debug mode',
                        action='store_true',
                        default=False)
#    parser.add_argument('-s', '--submit',
#                        help='submit the prediction',
#                        action='store_true',
#                        default=False)
    parser.add_argument('-m', '--message',
                        help='messages about the process',
                        type=str,
                        default='')

    args = parser.parse_args()
    sel_log(f'args: {sorted(vars(args).items())}', logger)
    return args


def parse_feature_args(logger=None):
    '''
    Policy
    ------------
    * experiment id must be required

    '''
    parser = argparse.ArgumentParser(
        prog='XXX.py',
        usage='ex) python XXX.py -f f001 -m "mk f001 features"',
        description='short explanation of args',
        add_help=True,
    )
    parser.add_argument('-f', '--feature_ids',
                        help='feature id',
                        type=str,
                        nargs='+',
                        required=True)
    parser.add_argument('-m', '--message',
                        help='messages about the process',
                        type=str,
                        default='')

    args = parser.parse_args()
    sel_log(f'args: {sorted(vars(args).items())}', logger)
    return args
