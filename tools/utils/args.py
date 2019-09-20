import argparse

from .logs import sel_log


def parse_args(logger=None):
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
    parser.add_argument('-e', '--exp_id',
                        help='experiment setting',
                        type=str,
                        required=True)
    parser.add_argument('-t', '--checkpoint',
                        help='the checkpoint u use',
                        type=str,
                        required=False,
                        default=None)
    parser.add_argument('-b', '--base_weight',
                        help='',
                        type=str,
                        required=False,
                        default=None)
    parser.add_argument('-m', '--message',
                        help='messages about the process',
                        type=str,
                        default='')
    parser.add_argument('-d', '--debug',
                        help='whether or not to use debug mode',
                        action='store_true',
                        default=False)
    parser.add_argument('-p', '--prediction',
                        help='flag which specifies prediction mode',
                        action='store_true',
                        default=False)
    parser.add_argument('-c', '--cell_types',
                        help='cell types',
                        nargs='+',
                        type=str,
                        default=['ALL'])

    args = parser.parse_args()
    sel_log(f'args: {sorted(vars(args).items())}', logger)
    return args


def parse_plate_args(logger=None):
    parser = argparse.ArgumentParser(
        prog='XXX.py',
        usage='ex) python -e e001 -d -m "e001, basic experiment"',
        description='short explanation of args',
        add_help=True,
    )
    parser.add_argument('-o', '--original_file',
                        help='the original file to which plate leak is applied.',
                        type=str,
                        required=True)
    args = parser.parse_args()
    sel_log(f'args: {sorted(vars(args).items())}', logger)
    return args
