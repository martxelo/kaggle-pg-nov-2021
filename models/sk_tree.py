import argparse

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

import mlflow

from utils import log_metric


def parse_args():

    parser = argparse.ArgumentParser(description='Train a decision tree classifier')

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
        '--criterion',
        metavar='string',
        type=str,
        default='gini',
        help='Criterion for measure the quality split')
    parser.add_argument(
        '--max_depth',
        metavar='int',
        type=int,
        default=None,
        help='Maximum depth of the tree')

    args = parser.parse_args()

    return args


def main():

    # arguments
    args = parse_args()
    nrows = args.nrows
    test_size = args.test_size
    criterion = args.criterion
    max_depth = args.max_depth
    
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