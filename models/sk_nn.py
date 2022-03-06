import argparse

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

import mlflow



def parse_args():

    parser = argparse.ArgumentParser(description='Train a neural network classifier')

    parser.add_argument(
        '--nrows',
        metavar='int',
        type=int,
        default=10000,
        help='Fraction of data for testing models')
    parser.add_argument(
        '--test_size',
        metavar='float',
        type=float,
        default=0.2,
        help='Select fraction of test data')
    parser.add_argument(
        '--n_components',
        metavar='float',
        type=float,
        default=0.95,
        help='Select number of components for PCA')
    parser.add_argument(
        '--hidden_layer_sizes',
        metavar='tuple',
        type=int,
        nargs='+',
        default=(50, 40, 3),
        help='Neurons in hidden layers')
    parser.add_argument(
        '--activation',
        metavar='string',
        type=str,
        default='logistic',
        help='Activation function for hidden layers')

    args = parser.parse_args()

    return args


def main():

    # arguments
    args = parse_args()
    nrows = args.nrows
    test_size = args.test_size
    n_components = args.n_components
    hidden_layer_sizes = tuple(args.hidden_layer_sizes)
    activation = args.activation

    if nrows < 0:
        nrows = None

    np.random.seed(42)

    # read data
    df = pd.read_csv('data/input/train.csv', nrows=nrows)
    df = df.drop(columns=['id'])
    
    # split in train and validation
    train, valid = train_test_split(df, test_size=test_size)

    # split in features and target
    x_train = train.drop(columns=['target'])
    y_train = train['target']
    x_valid = valid.drop(columns=['target'])
    y_valid = valid['target']

    with mlflow.start_run(run_name='Neural Net'):
        
        # steps for pipeline
        scaler = StandardScaler()
        pca = PCA(n_components=n_components)
        clf = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation)

        # create pipeline
        model = Pipeline(
            [('scaler', scaler),
             ('pca', pca),
             ('clf', clf)])

        # fit pipeline
        model.fit(x_train, y_train)

        # predict probability and labels
        pred_proba = model.predict_proba(x_valid)[:,1]
        pred_label = model.predict(x_valid)

        # calculate metrics
        auc = roc_auc_score(y_valid, pred_proba)
        acc = accuracy_score(y_valid, pred_label)
        f1 = f1_score(y_valid, pred_label)

        # log parameters and metrics
        mlflow.log_metrics({'auc': auc, 'acc': acc, 'f1': f1})
        mlflow.log_params({'pca_expl_var': n_components,
                           'pca_n_comp': model.steps[1][1].n_components_,
                           'pca_n_feat': model.steps[1][1].n_features_,
                           'clf_layers': hidden_layer_sizes,
                           'clf_activation': activation})

        # log model
        mlflow.sklearn.log_model(model, 'model')


if __name__ == '__main__':
    main()