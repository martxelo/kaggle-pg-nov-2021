import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

import mlflow

from .utils import log_metric


def train_sk_nn(args):

    # arguments
    nrows = args.nrows
    test_size = args.test_size
    n_components = args.n_components
    hidden_layer_sizes = tuple(args.hidden_layer_sizes)
    activation = args.activation

    # correct some arguments
    if nrows < 0:
        nrows = None

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

    # train model
    mlflow.set_experiment('SK nn')
    experiment = mlflow.get_experiment_by_name('SK nn')
    with mlflow.start_run(experiment_id=experiment.experiment_id):

        # fit pipeline
        model.fit(x_train, y_train)

        # predict probability
        pred_proba = model.predict_proba(x_valid)[:,1]

        # log metric
        log_metric(y_valid, pred_proba)

        # log parameters
        mlflow.log_params({
            'nrows': nrows,
            'pca_expl_var': n_components,
            'pca_n_comp': model.steps[1][1].n_components_,
            'pca_n_feat': model.steps[1][1].n_features_,
            'layers': hidden_layer_sizes,
            'activation': activation})

        # predict on test
        test = pd.read_csv('data/input/test.csv')
        test = test.drop(columns='id')
        test_proba = model.predict_proba(test)

        # write submission
        sam_sub = pd.read_csv('data/input/sample_submission.csv')
        sam_sub['target'] = test_proba[:,1]
        sam_sub.to_csv('data/output/sample_submission.csv', index=False)

        # log submission
        mlflow.log_artifact('data/output/sample_submission.csv')



if __name__ == '__main__':
    main()