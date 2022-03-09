import argparse

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

import mlflow

from .utils import log_metric


# global variables
INPUT_TRAIN = 'data/input/train.csv'
INPUT_TEST = 'data/input/test.csv'
INPUT_SUB = 'data/input/sample_submission.csv'
OUTPUT_SUB = 'data/output/sample_submission.csv'


def train_sk_tree(args):

    # arguments
    nrows = args.nrows
    test_size = args.test_size
    criterion = args.criterion
    max_depth = args.max_depth
    
    # correct some arguments
    if nrows < 0:
        nrows = None

    # read data
    df = pd.read_csv(INPUT_TRAIN)
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

    # classifier
    model = DecisionTreeClassifier(
        criterion=criterion,
        max_depth=max_depth)

    # train model
    mlflow.set_experiment('SK tree')
    experiment = mlflow.get_experiment_by_name('SK tree')
    with mlflow.start_run(run_name='Decision tree'):
        
        # fit classifier
        model.fit(x_train, y_train)

        # predict probability and labels
        pred_proba = model.predict_proba(x_valid)[:,1]
        
        # log metric
        log_metric(y_valid, pred_proba)

        # log parameters
        mlflow.log_params({
            'nrows': nrows,
            'criterion': criterion,
            'max_depth': max_depth})

        # predict on test
        test = pd.read_csv(INPUT_TEST)
        test = test.drop(columns='id')
        test_proba = model.predict_proba(test)

        # write submission
        sam_sub = pd.read_csv(INPUT_SUB)
        sam_sub['target'] = test_proba[:,1]
        sam_sub.to_csv(OUTPUT_SUB, index=False)

        # log submission
        mlflow.log_artifact(OUTPUT_SUB)
