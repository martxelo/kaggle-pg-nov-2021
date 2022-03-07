import argparse



def tf_nn_parser(subparsers):

    tf_nn = subparsers.add_parser(
        'tf_nn',
        help='Train TensorFlow model')

    tf_nn.add_argument(
        '--nrows',
        metavar='int',
        type=int,
        default=10000,
        help='Fraction of data for testing models')
    tf_nn.add_argument(
        '--test_size',
        metavar='float',
        type=float,
        default=0.2,
        help='Select fraction of test data')
    tf_nn.add_argument(
        '--n_components',
        metavar='float',
        type=float,
        default=0.95,
        help='Select number of components for PCA')
    tf_nn.add_argument(
        '--hidden_layer_sizes',
        metavar='tuple',
        type=int,
        nargs='+',
        default=(50, 40, 3),
        help='Neurons in hidden layers')
    tf_nn.add_argument(
        '--activation',
        metavar='string',
        type=str,
        default='logistic',
        help='Activation function for hidden layers')


def main():

    parser = argparse.ArgumentParser(description='Select a model to train')
    subparsers = parser.add_subparsers(dest='command')

    tf_nn_parser(subparsers)

    args = parser.parse_args()




if __name__ == '__main__':

    main()