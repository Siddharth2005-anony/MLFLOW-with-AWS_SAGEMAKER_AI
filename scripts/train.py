import os
import boto3
from io import BytesIO

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split,RandomizedSearchCV
from sklearn.metrics import (
    recall_score,
    precision_score,
    f1_score,
    roc_auc_score
)

from xgboost import XGBClassifier

import mlflow
import mlflow.xgboost
from mlflow.models import infer_signature

from itertools import product

#----------READING FROM S3----------------#
s3 = boto3.client("s3")
bucket_name = 'prod-sagemaker-ml'
key_name = 'data/processed'

def read_csv(filename:str):
    try:
        print("\n CONNECTING TO S3...\n")
        response = s3.get_object(
            Bucket = bucket_name,
            Key = f"{key_name}/{filename}"
        )

        data = pd.read_csv(
            BytesIO(response['Body'].read())
        )
        print("\n COMPLETED READING... \n")

    except Exception as e:
        return e

    return data

df = read_csv("train_bz_new.csv")

#----------DATASET------------------#

X = df.iloc[: , :-1]
Y = df.iloc[: , -1]

x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.2,random_state=42)

print(X.head())
print("\n BREAK \n")
print(Y.head())

#---------------MLFLOW---------------#
MLFLOW_APP_ARN = "arn:aws:sagemaker:eu-north-1:877231425975:mlflow-app/app-FFL7YSYGC73V"
mlflow.set_tracking_uri(MLFLOW_APP_ARN)
mlflow.set_experiment("SSD_DUMMY")


param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5],
    'learning_rate': [0.03, 0.05, 0.1],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

keys = param_grid.keys()

best_f1 = -1
best_model = None
best_params = None
best_run_id = None


print("\n CONNECTING TO MLFLOW...\n")
for values in product(*param_grid.values()):
    params = dict(zip(keys,values))

    with mlflow.start_run() as run:
        model = XGBClassifier(**params)

        model.fit(x_train,y_train)
        y_pred = model.predict(x_test)
        y_prob = model.predict_proba(x_test)[:, 1]

        recall = recall_score(y_test,y_pred)
        precision = precision_score(y_test,y_pred)
        f1 = f1_score(y_test,y_pred)
        roc = roc_auc_score(y_test,y_prob)

        print(f"\n recall: {recall} \n")
        print(f"\n precision: {precision} \n")
        print(f"\n f1: {f1} \n")
        print(f"\n roc: {roc} \n")

        #-----LOG PARAMS---------#
        mlflow.log_params(params)

        #----LOG METRICS----------#
        mlflow.log_metrics({
            "recall": recall,
            "precision": precision,
            "f1": f1,
            "roc": roc
        })

        #--------IDENTIFY BEST MODEL--------#
        if f1 > best_f1:
            best_f1 = f1
            best_model = model
            best_params = params.copy()
            best_run_id = run.info.run_id

#--------LOG ONLY THE BEST MODEL-----------#

print("\n========================================")
print("BEST MODEL")
print("========================================")
print(f"Best Params  : {best_params}")
print(f"Best Metrics      : {best_f1}")
print(f"Best Run ID  : {best_run_id}")

with mlflow.start_run(run_id=best_run_id):

    sign = infer_signature(
        x_train,
        best_model.predict(x_train)
    )

    model_info = mlflow.xgboost.log_model(
        xgb_model=best_model,
        name="model",
        signature=sign,
        input_example=x_train.head(3)
    )

    mlflow.set_tag("model_status","best")
    mlflow.log_metric("best_f1",best_f1)


print("\nBEST MODEL SUCCESSFULLY LOGGED TO SAGEMAKER MLFLOW")

