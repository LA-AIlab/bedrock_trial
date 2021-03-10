#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import json
import pickle
import pandas
import numpy as np
from bedrock_client.bedrock.metrics.service import ModelMonitoringService
from flask import Flask, Response, current_app, request
# from trainn import X_val,y_val
# from train_bedrock import data
from constants import FEATURE_COLS

OUTPUT_MODEL_NAME = "/artefact/lgb_model.pkl"


f="gs://bucket-bedrock/features_bedrock.csv"
df = pandas.read_csv(f)
print("print values and type for df...:", type(df), df)

feat=df[FEATURE_COLS]
feat=feat.fillna(0)
print("values for feat", type(feat))
def predict_prob(feat,model=pickle.load(open(OUTPUT_MODEL_NAME, "rb"))):
    """Predict churn probability given subscriber_features.
    Args:
        subscriber_features (dict)
        model
    Returns:
        prob (float): churn probability
    """
    
    # Score
    prob = (model.predict_proba(np.array(feat).reshape(1, -1))[:, 1].item())

    # Log the prediction
    # Log the prediction
    current_app.monitor.log_prediction(
        request_body=json.dumps(feat),
        features=feat.values[0],
        output=prob)
    
    return prob

app = Flask(__name__)
@app.route("/", methods=["POST"])
def get_churn():
    """Returns the `churn_prob` given the subscriber features"""

    feat = request.json
    result = {
        "prob": predict_prob(feat)
    }
    return result



@app.before_first_request
def init_background_threads():
    """Global objects with daemon threads will be stopped by gunicorn --preload flag.
    So instantiate them here instead.
    """
    current_app.monitor = ModelMonitoringService()


@app.route("/metrics", methods=["GET"])
def get_metrics():
    """Returns real time feature values recorded by prometheus
    """
    body, content_type = current_app.monitor.export_http(
        params=request.args.to_dict(flat=False),
        headers=request.headers,
    )
    return Response(body, content_type=content_type)


def main():
    """Starts the Http server"""
    app.run()


if __name__ == "__main__":
    main()

