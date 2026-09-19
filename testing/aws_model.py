from fastapi import FastAPI
import boto3
from pydantic import BaseModel
from typing import List

app = FastAPI()

runtime = boto3.client(
    "sagemaker-runtime",
    region_name="eu-north-1"
)


class ModelInput(BaseModel):
    data: List[int]


@app.get("/test_site")
def response_in():
    return {"msg": "hello"}


@app.post("/test_model")
def load_input(data: ModelInput):

    # Convert Python list → CSV string
    payload = ",".join(map(str, data.data))

    # Invoke SageMaker endpoint
    response = runtime.invoke_endpoint(
        EndpointName="xgb-1909",
        ContentType="text/csv",
        Body=payload.encode("utf-8")
    )

    # Read prediction
    prediction = float(
        response["Body"].read().decode("utf-8")
    )

    # Convert probability → class
    label = int(prediction >= 0.5)

    return {
        "Predicted class": label,
        "msg": ["No Failure" if label == 0 else "Failure"]
    }