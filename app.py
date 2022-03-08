import argparse


def tf_nn_parser(subparsers):

    tf_nn = subparsers.add_parser(
        'tf_nn',
        help='Train a TensorFlow MLP model')

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
        default=[50, 40, 3],
        help='Neurons in hidden layers')
    tf_nn.add_argument(
        '--activation',
        metavar='string',
        type=str,
        default='relu',
        help='Activation function for hidden layers')
    tf_nn.add_argument(
        '--stddev',
        metavar='float',
        type=float,
        default=0.01,
        help='Gaussian noise for not overfitting')
    tf_nn.add_argument(
        '--epochs',
        metavar='int',
        type=int,
        default=200,
        help='Number of epochs')
    tf_nn.add_argument(
        '--batch_size',
        metavar='int',
        type=int,
        default=1024,
        help='Batch_size for fitting the model')


def sk_nn_parser(subparsers):

    sk_nn = subparsers.add_parser(
        'sk_nn',
        help='Train a SKlearn MLPClassifier model')

    sk_nn.add_argument(
        '--nrows',
        metavar='int',
        type=int,
        default=10000,
        help='Fraction of data for testing models')
    sk_nn.add_argument(
        '--test_size',
        metavar='float',
        type=float,
        default=0.2,
        help='Select fraction of test data')
    sk_nn.add_argument(
        '--n_components',
        metavar='float',
        type=float,
        default=0.95,
        help='Select number of components for PCA')
    sk_nn.add_argument(
        '--hidden_layer_sizes',
        metavar='tuple',
        type=int,
        nargs='+',
        default=(50, 40, 3),
        help='Neurons in hidden layers')
    sk_nn.add_argument(
        '--activation',
        metavar='string',
        type=str,
        default='logistic',
        help='Activation function for hidden layers')


def sk_tree_parser(subparsers):

    sk_tree = subparsers.add_parser(
        'sk_tree',
        help='Train a SKlearn DecisionTreeClassifier model')

    sk_tree.add_argument(
        '--nrows',
        metavar='int',
        type=int,
        default=10000,
        help='Fraction of data for testing models')
    sk_tree.add_argument(
        '--test_size',
        metavar='float',
        type=float,
        default=0.2,
        help='Select fraction of test data')
    sk_tree.add_argument(
        '--criterion',
        metavar='string',
        type=str,
        default='gini',
        help='Criterion for measure the quality split')
    sk_tree.add_argument(
        '--max_depth',
        metavar='int',
        type=int,
        default=None,
        help='Maximum depth of the tree')


def main():

    # main parser and subparsers
    parser = argparse.ArgumentParser(description='Select a model to train')
    subparsers = parser.add_subparsers(dest='command')

    # add subparsers
    tf_nn_parser(subparsers)
    sk_nn_parser(subparsers)
    sk_tree_parser(subparsers)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
    elif args.command == 'tf_nn':
        from models.tf_nn import train_tf_nn
        train_tf_nn(args)
    elif args.command == 'sk_nn':
        from models.sk_nn import train_sk_nn
        train_sk_nn(args)
    elif args.command == 'sk_tree':
        from models.sk_tree import train_sk_tree
        train_sk_tree(args)
        

if __name__ == '__main__':

    main()