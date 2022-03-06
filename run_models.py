'''
Example for running many models with different parameters
'''

import os
from itertools import product




hidden_layer_sizes = [
    [50, 40, 10],
    [80, 30, 5],
    [40, 40, 40],
    [80, 60, 40, 30, 20, 10],
    [10, 10, 10, 10, 10, 10, 10, 10]]
activation = ['relu', 'tanh', 'swish', 'sigmoid', 'elu']
test_size = [0.10, 0.20]
n_components = [100, 0.99, 0.95, 0.90, 0.50]
stddev = [0.000, 0.010, 0.025, 0.030, 0.050]
batch_size = [512, 1024, 2048]

for hls, ac, ts, nc, std, bs in product(hidden_layer_sizes,
                                        activation,
                                        test_size,
                                        n_components,
                                        stddev,
                                        batch_size):
    
    command = 'python models/tf_nn.py --nrows -1'
    command += ' --hidden_layer_sizes ' + ' '.join([str(i) for i in hls])
    command += ' --activation ' + ac
    command += ' --test_size ' + str(ts)
    command += ' --n_components ' + str(nc)
    command += ' --stddev ' + str(std)
    command += ' --batch_size ' + str(bs)
    os.system(command)
    break
