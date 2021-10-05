from configs import TRAIN_CONFIG, ENC_CONFIG
import argparse
import pprint

def add_common_args(parser):
    # --------------------------- Model setup args ------------------------
    parser.add_argument('--block_type', '-B', type=str, nargs='?', default=config['block_type'],
                    help='The block type that will be used, one of "graph_wavenet", "graph_wavenet_original", '
                         '"graph_wavenet_original_te", defaults to "%s".' % config['block_type'])
    parser.add_argument('--n_blocks', '-N', type=int, nargs='?', default=config['n_blocks'],
                    help='The number of blocks, defaults to %d.' % config['n_blocks'])
    parser.add_argument('--filters', '-F', type=int, nargs='?', default=config['filters'][0],
                    help='The number of kernels used in every block, defaults to %d.' % config['filters'][0])
    parser.add_argument('--skip_channels', '-S', type=int, nargs='?', default=config['skip_channels'],
                    help='Number of output channels (skip channels) of each block, defaults to %d.' % config['skip_channels'])
    parser.add_argument('--abs', '-A', action='store_true', default=not config['from_velocities'],
                    help='If set, the model inputs are absolute quaternions instead of velocities '
                         '%s' % ('(default)' if not config['from_velocities'] else ''))
    parser.add_argument('--vel', '-V', action='store_true', default=config['from_velocities'],
                    help='If set, the model inputs are velocity quaternions instead of absolute quaternions ' 
                         '%s' % ('(default)' if config['from_velocities'] else ''))
    parser.add_argument('--checkpoint', '-C', type=str, nargs='?',
                    help='A directory containing a checkpoint to load the model weights from (if given, '
                         'continues training from that checkpoint).')

    # ------------------------- Training setup args -----------------------
    parser.add_argument('--batch_size', '-b', type=int, nargs='?', default=config['batch_size'],
                    help='Batch size, defaults to %d.' % config['batch_size'])
    parser.add_argument('--condition_length', '-c', type=int, nargs='?', default=config['condition_length'],
                    help='The number of input timesteps, defaults to %d.' % config['condition_length'])
    parser.add_argument('--forecasting_length', '-f', type=int, nargs='?', default=config['forecasting_length'],
                    help='The number of output timesteps, defaults to %d.' % config['forecasting_length'])

    # ------------------------- Dataset setup args ------------------------
    parser.add_argument('--data_dir', '-d', type=str, nargs='?', default=config['data_dir'],
                    help='Directory of the dataset. This directory should contain the folders for the individual '
                         'actors S1, S5, S6, ...., defaults to "%s".' % config['data_dir'])

    # ---------------------------- Misc  args -----------------------------
    parser.add_argument('--output_dir', '-o', type=str, nargs='?', default=config['output_dir'],
                    help='The directory that will contain all training and evaluation artifacts, '
                         'defaults to "%s"' % config['output_dir'])
    return parser

def train(**kwargs):
    '''
    Train a forecasting model on H3.6M train- and validation-set.
    '''
    from train import train
    pp = pprint.PrettyPrinter(indent=4)
    config_str = pp.pformat(kwargs)
    print('Training config:')
    print(config_str)
    train(**kwargs)

def eval(**kwargs):
    '''
    Evaluate a forecasting model on H3.6M test-set.
    '''
    from evaluate import evaluate
    pp = pprint.PrettyPrinter(indent=4)
    config_str = pp.pformat(kwargs)
    print('Evaluation config:')
    print(config_str)
    evaluate(**kwargs)

if __name__ == '__main__':
    config = {}
    config.update(ENC_CONFIG)
    config.update(TRAIN_CONFIG)

    parser = argparse.ArgumentParser(add_help=True,
        description='Apply the lightweight autoregressive model to the H3.6M dataset.')
    parent = argparse.ArgumentParser(add_help=False)
    subparsers = parser.add_subparsers(help='Desired mode, either "train" or "eval".')
    train_parser = subparsers.add_parser('train',
        description='Train a lightweight autoregressive model on the H3.6M dataset.')
    train_parser.set_defaults(func=train)
    eval_parser = subparsers.add_parser('eval',
        description='Evaluate a lightweight autoregressive model on the H3.6M dataset.')
    eval_parser.set_defaults(func=eval)
    
    train_parser = add_common_args(train_parser)
    train_group = train_parser.add_argument_group('training arguments')
    eval_parser = add_common_args(eval_parser)
    eval_group = eval_parser.add_argument_group('evaluation arguments')

    train_group.add_argument('--epochs', '-e', type=int, nargs='?', default=config['epochs'],
                    help='Number of training epochs, defaults to %d.' % config['epochs'])
    eval_group.add_argument('--eval_protocol', '-p', type=str, nargs='?', default='std',
                    help='Evaluation protocol to use, defaults to "std".')

    args = parser.parse_args()
    config.update(vars(args))
    config['filters'] = [config['filters']] * config['n_blocks']
    pprint.pprint(config)

    if not config['abs'] != config['vel']:
        raise Exception('The model can only predict either absolute quaternions OR relative quaternions, '
                        'i.e. velocities. Set only one of --abs, --vel.')

    config['from_velocities'] =  not config['abs']

    args.func(**config)