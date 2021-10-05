import os

ENC_CONFIG = {
    # specifies the block type of the forecasting model
    # one of:
    #   graph_wavenet_original : The Block that GraphWavenet uses
    #   graph_wavenet          : Our modified graph Graph-Wavenet block; uses TE Convolution and K-GCN
    #   graph_wavenet_te       : Our modified graph Graph-Wavenet block that uses the TE Convolution
    #                            (ours) and diffusion graph-convolution from original Graph-Wavenet
    'block_type'        : 'graph_wavenet', # 'graph_wavenet_original', 'graph_wavenet_original_te'
    'selfloops'         : True,
    'n_blocks'          : 5,
    'dropout_rate'      : 0.,
    'filters'           : [64, 64, 64, 64, 64],
    'kernel_sizes'      : [2]*5,
    'dilations'         : [ 2**i for i in range(5) ],
    'skip_channels'     : 256,
    'mlp_depth'         : 1, # not used
    'out_channels'      : 4, # 4 quaternion dimensions
}

TRAIN_CONFIG = {
    'init_learning_rate' : 1e-3,
    'condition_length'   : 32,
    'forecasting_length' : 10,
    'from_velocities'    : True,
    'batch_size'         : 16,
    'epochs'             : 3000,
    'data_dir'           : 'data/h3.6m',
    'output_dir'         : './results'
}

# number of consecutive timesteps to sample from the dataset
TRAIN_CONFIG['time_horizon'] = TRAIN_CONFIG['condition_length'] + \
                               TRAIN_CONFIG['forecasting_length']


def save(config, path):
    conf = dict()
    conf['encoder'] = ENC_CONFIG
    conf['train'] = TRAIN_CONFIG

    out_str = ''

    header = '-'*20 + ' %s ' + '-'*20 + '\n'
    content = '  %-15s%s\n'

    for k, v in conf.items():
        out_str += header % k.upper()

        for attr, c in v.items():
            out_str += content % (attr, str(c))

    file = os.path.join(path, 'configs.txt')
    
    if not os.path.exists(path):
        os.makedirs(path)

    with open(file, 'w') as f:
        f.write(out_str)