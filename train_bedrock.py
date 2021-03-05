#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import logging
import os
import pickle
import base64
from google.cloud import storage

import numpy as np
import pandas as pd
import lightgbm as lgb
from bedrock_client.bedrock.analyzer.model_analyzer import ModelAnalyzer
from bedrock_client.bedrock.analyzer import ModelTypes
from bedrock_client.bedrock.api import BedrockApi
from bedrock_client.bedrock.metrics.service import ModelMonitoringService
from sklearn import metrics
from sklearn.model_selection import train_test_split
from constants import FEATURE_COLS, TARGET_COL
from preprocess import util

# Set environment variable
# os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "my_secret.json"
# api_key=os.getenv("SECRET_KEY_1")

# Retrieve secret value from environment variable
secret_value = os.getenv("SECRET_KEY_1")
print("type of var secret value.......: ",type(secret_value))

# Decode the string to a file using the complementary method as encoding
decoded_string = base64.b64decode(secret_value)
with open("service_account.json", "wb") as sa_file:  # You may have to open the file with "w" depending on how you read it when encoding
    sa_file.write(decoded_string) # wb means binary mode for writing
    
# Explicitly use service account credentials by specifying the private key file.
storage_client = storage.Client.from_service_account_json("service_account.json")

# Make an authenticated API request
buckets = list(storage_client.list_buckets())
print("are we getting into buckets?")
print(buckets)


blob = bucket.blob('features_bedrock.csv')
path = "gs://bucket-bedrock/features_bedrock.csv"
data = pd.read_csv(path)

# TEMP_DATA_BUCKET="gs://bucket-bedrock/features_bedrock.csv" #"gs://student_bucket"
# data=util.load_data(TEMP_DATA_BUCKET, storage_options = service_account.json)
data = data.fillna(0)
print(data.head())



FEATURES_DATA = os.path.join(os.getenv("TEMP_DATA_BUCKET"),
                             os.getenv("FEATURES_DATA"))
LR = float(os.getenv("LR"))
NUM_LEAVES = int(os.getenv("NUM_LEAVES"))
N_ESTIMATORS = int(os.getenv("N_ESTIMATORS"))
OUTPUT_MODEL_NAME = os.getenv("OUTPUT_MODEL_NAME")


def compute_log_metrics(clf, x_val, y_val):
    """Compute and log metrics."""
    print("\tEvaluating using validation data")
    y_prob = clf.predict_proba(x_val)[:, 1]
    y_pred = (y_prob > 0.5).astype(int)

    acc = metrics.accuracy_score(y_val, y_pred)
    precision = metrics.precision_score(y_val, y_pred)
    recall = metrics.recall_score(y_val, y_pred)
    f1_score = metrics.f1_score(y_val, y_pred)
    roc_auc = metrics.roc_auc_score(y_val, y_prob)
    avg_prc = metrics.average_precision_score(y_val, y_prob)

    print(f"Accuracy          = {acc:.6f}")
    print(f"Precision         = {precision:.6f}")
    print(f"Recall            = {recall:.6f}")
    print(f"F1 score          = {f1_score:.6f}")
    print(f"ROC AUC           = {roc_auc:.6f}")
    print(f"Average precision = {avg_prc:.6f}")

    # Log metrics
    bedrock = BedrockApi(logging.getLogger(__name__))
    bedrock.log_metric("Accuracy", acc)
    bedrock.log_metric("Precision", precision)
    bedrock.log_metric("Recall", recall)
    bedrock.log_metric("F1 score", f1_score)
    bedrock.log_metric("ROC AUC", roc_auc)
    bedrock.log_metric("Avg precision", avg_prc)
    bedrock.log_chart_data(y_val.astype(int).tolist(),
                           y_prob.flatten().tolist())

    # Calculate and upload xafai metrics
    analyzer = ModelAnalyzer(clf, 'tree_model', model_type=ModelTypes.TREE).test_features(x_val)
    analyzer.test_labels(y_val.values).test_inference(y_pred)
    analyzer.analyze()


def main():
    """Train pipeline"""
    model_data = pd.read_csv(FEATURES_DATA)

    print("\tSplitting train and validation data")
    x_train, x_val, y_train, y_val = train_test_split(
        model_data[FEATURE_COLS],
        model_data[TARGET_COL],
        test_size=0.2,
    )

    print("\tTrain model")
    clf = lgb.LGBMClassifier(
        num_leaves=NUM_LEAVES,
        learning_rate=LR,
        n_estimators=N_ESTIMATORS,
    )
    clf.fit(x_train, y_train)
    compute_log_metrics(clf, x_val, y_val)

    print("\tComputing metrics")
    selected = np.random.choice(model_data.shape[0], size=1000, replace=False)
    features = model_data[FEATURE_COLS].iloc[selected]
    inference = clf.predict_proba(features)[:, 1]

    ModelMonitoringService.export_text(
        features=features.iteritems(),
        inference=inference.tolist(),
    )

    print("\tSaving model")
    with open("/artefact/" + OUTPUT_MODEL_NAME, "wb") as model_file:
        pickle.dump(clf, model_file)


if __name__ == "__main__":
    main()

