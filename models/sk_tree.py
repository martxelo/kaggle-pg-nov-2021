import argparse

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

import mlflow



def parse_args():

    parser = argparse.ArgumentParser(description='Train a decision tree classifier')

    parser.add_argument(
        '-test_size',
        metavar='float',
        type=float,
        default=0.2,
        help='Select fraction of test data'
    )
    parser.add_argument(
        '-criterion',
        metavar='string',
        type=str,
        default='gini',
        help='Criterion for measure the quality split'
    )
    parser.add_argument(
        '-max_depth',
        metavar='int',
        type=int,
        default=None,
        help='Maximum depth of the tree'
    )

    args = parser.parse_args()

    test_size = args.test_size
    criterion = args.criterion
    max_depth = args.max_depth

    return test_size, criterion, max_depth



def main():

    test_size, criterion, max_depth = parse_args()

    np.random.seed(42)

    # read data
    df = pd.read_csv('data/input/train.csv', nrows=5000)
    df = df.drop(columns=['id'])
    
    # split in train and validation
    train, valid = train_test_split(df, test_size=test_size)

    # split in features and target
    x_train = train.drop(columns=['target'])
    y_train = train['target']
    x_valid = valid.drop(columns=['target'])
    y_valid = valid['target']

    with mlflow.start_run(run_name='Decision trees'):
        
        # classifier
        clf = DecisionTreeClassifier(
            criterion=criterion,
            max_depth=max_depth,
        )

        # fit classifier
        clf.fit(x_train, y_train)

        # predict probability and labels
        pred_proba = clf.predict_proba(x_valid)[:,1]
        pred_label = clf.predict(x_valid)

        # calculate metrics
        auc = roc_auc_score(y_valid, pred_proba)
        acc = accuracy_score(y_valid, pred_label)
        f1 = f1_score(y_valid, pred_label)

        # log parameters and metrics
        mlflow.log_metrics({'auc': auc, 'acc': acc, 'f1': f1})
        mlflow.log_params({'criterion': criterion,
                           'max_depth': max_depth
        })

        # log model
        mlflow.sklearn.log_model(clf, 'Decision trees')


if __name__ == '__main__':
    main()