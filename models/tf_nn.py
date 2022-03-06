import argparse

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import tensorflow.keras.layers as L
from tensorflow.keras import Model
from tensorflow.keras.metrics import AUC
from tensorflow.keras.callbacks import EarlyStopping

import mlflow

from utils import log_metric



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
        default=[50, 40, 3],
        help='Neurons in hidden layers')
    parser.add_argument(
        '--activation',
        metavar='string',
        type=str,
        default='relu',
        help='Activation function for hidden layers')
    parser.add_argument(
        '--stddev',
        metavar='float',
        type=float,
        default=0.01,
        help='Gaussian noise for not overfitting')
    parser.add_argument(
        '--batch_size',
        metavar='int',
        type=int,
        default=1024,
        help='Batch_size for fitting the model')

    args = parser.parse_args()

    return args


def tf_model(shape, hidden_layer_sizes, activation, stddev):

    inputs = L.Input(shape)

    # some gaussian noise
    outputs = L.GaussianNoise(stddev)(inputs)

    # hidden layers
    for neurons in hidden_layer_sizes:
        outputs = L.Dense(neurons, activation=activation)(outputs)

    # last layer has 'sigmoid' activation function
    outputs = L.Dense(1, activation='sigmoid')(outputs)

    # build model
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=[AUC(), 'acc'])

    return model


def main():

    # arguments
    args = parse_args()
    nrows = args.nrows
    test_size = args.test_size
    n_components = args.n_components
    hidden_layer_sizes = args.hidden_layer_sizes
    activation = args.activation
    stddev = args.stddev
    batch_size = args.batch_size

    # correct some arguments
    if nrows < 0:
        nrows = None
    if n_components >= 1:
        n_components = int(n_components)

    # read data
    df = pd.read_csv('data/input/train.csv')
    df = df.drop(columns=['id'])

    # subsample
    if nrows is not None:
        df = df.sample(nrows)
    
    # split in train and validation
    train, valid = train_test_split(df, test_size=test_size)

    # split in features and target
    x_train = train.drop(columns=['target'])
    y_train = train['target']
    x_valid = valid.drop(columns=['target'])
    y_valid = valid['target']

    # scale data
    scaler = StandardScaler().fit(x_train)
    x_train = scaler.transform(x_train)
    x_valid = scaler.transform(x_valid)
    
    # PCA
    pca = PCA(n_components=n_components).fit(x_train)
    x_train = pca.transform(x_train)
    x_valid = pca.transform(x_valid)

    # get model
    model = tf_model(
        x_train.shape[1],
        hidden_layer_sizes,
        activation,
        stddev)

    # callbacks
    callbacks = [EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True)]

    # train model
    mlflow.set_experiment('TF nn')
    experiment = mlflow.get_experiment_by_name('TF nn')
    with mlflow.start_run(experiment_id=experiment.experiment_id):

        # fit pipeline
        model.fit(
            x_train,
            y_train,
            epochs=500,
            batch_size=batch_size,
            validation_data=(x_valid, y_valid),
            callbacks=callbacks)

        # predict probability
        pred_proba = model.predict(x_valid)

        # log metric
        log_metric(y_valid, pred_proba)

        # log parameters
        pca_expl_var = min(1.0, pca.explained_variance_ratio_.sum())
        mlflow.log_params({
            'nrows': nrows,
            'stddev': stddev,
            'pca_expl_var': pca_expl_var,
            'pca_n_comp': pca.n_components_,
            'layers': hidden_layer_sizes,
            'activation': activation})

        # predict on test
        test = pd.read_csv('data/input/test.csv')
        test = test.drop(columns='id')
        test = scaler.transform(test)
        test = pca.transform(test)
        test_proba = model.predict(test)

        # write submission
        sam_sub = pd.read_csv('data/input/sample_submission.csv')
        sam_sub['target'] = test_proba.reshape(-1)
        sam_sub.to_csv('data/output/sample_submission.csv', index=False)

        # log submission
        mlflow.log_artifact('data/output/sample_submission.csv')



if __name__ == '__main__':
    main()