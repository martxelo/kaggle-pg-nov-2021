
from sklearn.metrics import roc_auc_score, roc_curve

import mlflow

import plotly.express as px


def log_metric(y_valid, pred_proba):

    # calculate metrics
    auc = roc_auc_score(y_valid, pred_proba)

    # log and metrics
    mlflow.log_metric('auc', auc)

    # roc curve
    fpr, tpr, _ = roc_curve(y_valid, pred_proba)

    # log plot
    fig = px.line(
        x=fpr,
        y=tpr,
        title='AUC = ' + str('%.4f' %auc),
        labels={'x': 'fpr', 'y': 'tpr'})
    fig.update_layout(
        autosize=False,
        width=500,
        height=500)
    mlflow.log_figure(fig, 'auc.html')
