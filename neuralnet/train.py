import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score

import mlflow





def main():

    np.random.seed(42)

    # read data
    df = pd.read_csv('data/input/train.csv', nrows=5000)
    
    # split in train and validation
    train, valid = train_test_split(df, test_size=0.2)

    # split in features and target
    x_train = train.drop(columns=['target'])
    y_train = train['target']
    x_valid = valid.drop(columns=['target'])
    y_valid = valid['target']
    

    with mlflow.start_run():

        model = MLPClassifier(
            hidden_layer_sizes=(40, 10),
            activation='logistic'
        )
        model.fit(x_train, y_train)

        pred = model.predict_proba(x_valid)[:,1]

        score = roc_auc_score(y_valid, pred)

        mlflow.log_metrics({'AUC': score})

        mlflow.sklearn.log_model(model, 'model')



    





if __name__ == '__main__':
    main()