import os
import time
from datetime import datetime, timedelta
# from dateutil import parser

import matplotlib.pyplot as plt
from sklearn import metrics

import pytz
import pandas as pd
# from contextlib import contextmanager
from sklearn.preprocessing import OneHotEncoder


# @contextmanager
# def timer(title):
#     t0 = time.time()
#     yield
#     print("  Time taken for {} = {:.0f}s".format(title, time.time() - t0))
    

# def get_execution_date():
#     """Get execution date using server time."""
#     execution_date = os.getenv("EXECUTION_DATE")

#     # Run DAG daily at 01H00 UTC = 09H00 SGT
#     if not execution_date:
#         dt_now = datetime.now(pytz.utc) - timedelta(days=1)
#         # If time now is between 00H00 UTC and 01H00 UTC, set execution date 1 more day earlier
#         if dt_now.strftime("%H:%M") < "01:00":
#             dt_now = dt_now - timedelta(days=1)
#         execution_date = parser.parse(dt_now.strftime("%Y-%m-%d"))
#     else:
#         execution_date = parser.parse(execution_date[:10])

#     return execution_date


def load_data(file_path, file_type="pd_csv"):
    """Load data."""
    
    if file_type == "pd_csv":
        return pd.read_csv(file_path)
    elif file_type == "pd_parquet":
        return pd.read_parquet(file_path)
    else:
        raise Exception("Not implemented")


def onehot_enc(df, categorical_columns, categories):
    """One-hot encoding of categorical columns."""
    noncategorical_cols = [col for col in df.columns if col not in categorical_columns]
    
    enc = OneHotEncoder(categories=categories,
                        sparse=False,
                        handle_unknown='ignore')
    y = enc.fit_transform(df[categorical_columns].fillna("None"))
    
    ohe_cols = [
        f"{col}_{c}" for col, cats in zip(categorical_columns, categories) for c in cats]
    df1 = pd.DataFrame(y, columns=ohe_cols)
    
    output_df = pd.concat([df[noncategorical_cols], df1], axis=1)
    return output_df, ohe_cols

def lgb_roc_auc_score(y_true, y_pred):
    return "roc_auc", metrics.roc_auc_score(y_true, y_pred), True


def print_results(actual, probs):
    preds = (probs > 0.5).astype(int)
    print('Confusion matrix:')
    print(metrics.confusion_matrix(actual, preds), "\n")
    print(metrics.classification_report(actual, preds))


# ROC(tpr-fpr) curve
def plot_roc_curve(actual, pred, ax=None):
    """Plot ROC."""
    fpr, tpr, _ = metrics.roc_curve(actual, pred)

    if ax is None:
        fig, ax = plt.subplots()
        
    ax.plot(fpr, tpr)
    ax.plot([0, 1], [0, 1], linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC AUC = {:.4f}'.format(
        metrics.roc_auc_score(actual, pred)))
    return ax


# Precision-recall curve
def plot_pr_curve(actual, pred, ax=None):
    """Plot PR curve."""
    precision, recall, _ = metrics.precision_recall_curve(actual, pred)

    if ax is None:
        fig, ax = plt.subplots()
        
    ax.step(recall, precision, color='b', alpha=0.2, where='post')
    ax.fill_between(recall, precision, alpha=0.2, color='b', step='post')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Avg precision = {:.4f}'.format(
        metrics.average_precision_score(actual, pred)))
    return ax
