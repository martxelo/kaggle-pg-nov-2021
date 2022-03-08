'''
Example for running many models with different parameters
'''

import os
from itertools import product

# parameters
nrows = [-1]
hidden_layer_sizes = [
    [80, 60, 40, 30, 20, 10],
    [10, 10, 10, 10, 10, 10, 10, 10]]
activation = ['relu', 'tanh', 'swish', 'sigmoid']
test_size = [0.20]
n_components = [100]
stddev = [0.025]
batch_size = [2048]
epochs = [200]


# run all combinations
for nr, hls, ac, ts, nc, std, bs, ep in product(nrows,
                                        hidden_layer_sizes,
                                        activation,
                                        test_size,
                                        n_components,
                                        stddev,
                                        batch_size,
                                        epochs):
    
    command = 'python app.py tf_nn'
    command += ' --nrows ' + str(nr)
    command += ' --hidden_layer_sizes ' + ' '.join([str(i) for i in hls])
    command += ' --activation ' + ac
    command += ' --test_size ' + str(ts)
    command += ' --n_components ' + str(nc)
    command += ' --stddev ' + str(std)
    command += ' --batch_size ' + str(bs)
    command += ' --epochs ' + str(ep)
    print(command)
    os.system(command)
