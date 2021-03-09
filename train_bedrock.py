#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import logging
import os
import pickle
import base64
import gcsfs
from io import BytesIO
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
print("")

# Decode the string to a file using the complementary method as encoding
decoded_string = base64.b64decode(secret_value)
print("type of decoded string....",type(decoded_string))
print("")
with open("service_account.json", "wb") as sa_file:  # You may have to open the file with "w" depending on how you read it when encoding
    sa_file.write(decoded_string) # wb means binary mode for writing
    
# Explicitly use service account credentials by specifying the private key file.
storage_client = storage.Client.from_service_account_json("service_account.json")

# Make an authenticated API request
# bucket = storage_client.get_bucket('bucket-bedrock')
# blob = bucket.blob('features_bedrock.csv')

# bucket = storage_client.bucket('bucket-bedrock')
# blobs = bucket.list_blobs()
# for blob in blobs:
#     print(blob.name)

# buckets = list(storage_client.list_buckets())
# print("are we getting into buckets?")
# print(buckets[0])
# # blob = buckets.blob('features_bedrock.csv')


# buckets_list = list(storage_client.list_buckets())
# print("are we getting into buckets?")
# bucket_name='bucket-bedrock'
# print("bucket_name variable value is: ",bucket_name)
# print("")
# bucket = storage_client.bucket(bucket_name)
# blobs = bucket.list_blobs()
# print("printing values in blobs....",blobs)
# print("type of blob",type(blobs))

# list_temp_raw = []
# for file in blobs:
#     filename = file.name
#     temp = pd.read_csv('gs://'+'bucket-bedrock'+'/'+'features_bedrock.csv', encoding='utf-8')
#     print(filename, temp.head())
#     list_temp_raw.append(temp)

# data = pd.concat(list_temp_raw)
print("now we have data: and putting values of data in temp data bucket")
# print("type of data variable is....: ",type(data))


# TEMP_DATA_BUCKET = data
# print("value for temp data bucket is: ",TEMP_DATA_BUCKET)
print("did u get the tmp data bucket values?...")
# data = pd.read_csv(TEMP_DATA_BUCKET)

TEMP_DATA_BUCKET="gs://bucket-bedrock/features_bedrock.csv" #"gs://student_bucket"
print("TEMP_DATA_BUCKET type is : ",type(TEMP_DATA_BUCKET))
print("TEMP_DATA_BUCKET is: ",TEMP_DATA_BUCKET)
# data=util.load_data(TEMP_DATA_BUCKET, storage_options = service_account.json)
# fs = gcsfs.GCSFileSystem(project='mybedrock-trial')
# with fs.open('gs://bucket-bedrock/features_bedrock.csv') as f:
#     data = pd.read_csv(f)

# TEMP_DATA_BUCKET
print("type of data of TEMP_DATA_BUCKET:.....",type(TEMP_DATA_BUCKET))
data=TEMP_DATA_BUCKET
data = data.fillna(0)
print(data.head())



FEATURES_DATA = data.iloc[:,:20]
print("type of features_data", type(FEATURES_DATA))
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
    #model_data = pd.read_csv(FEATURES_DATA)

    print("\tSplitting train and validation data")
    x_train, x_val, y_train, y_val = train_test_split(
        data[FEATURE_COLS],
        data[TARGET_COL],
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
    selected = np.random.choice(data.shape[0], size=1000, replace=False)
    features = data[FEATURE_COLS].iloc[selected]
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

