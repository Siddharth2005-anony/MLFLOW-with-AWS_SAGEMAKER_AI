# Server Failure Prediction - MLOps Project

An end-to-end MLOps pipeline for predicting server/hard drive failures using SMART (Self-Monitoring, Analysis and Reporting Technology) attributes. The project uses XGBoost for binary classification and deploys the model on AWS SageMaker with a FastAPI inference service.

## Overview

This project predicts potential server/hard drive failures by analyzing SMART attributes collected from drives. The model is trained using XGBoost, tracked with MLflow, deployed to AWS SageMaker, and served via a FastAPI REST API.

## Architecture

```
┌─────────────┐     ┌─────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Raw Data   │────▶│  Training   │────▶│   MLflow         │────▶│  AWS SageMaker  │
│  (CSV)      │     │  (XGBoost)  │     │  Tracking        │     │  Endpoint       │
└─────────────┘     └─────────────┘     └──────────────────┘     └────────┬────────┘
                                                                          │
                                                                          ▼
                                                                 ┌─────────────────┐
                                                                 │   FastAPI       │
                                                                 │   Inference API │
                                                                 └─────────────────┘
```

## Features

- **Data**: SMART attributes from hard drives (January–March 2020)
- **Model**: XGBoost binary classifier (Failure / No Failure)
- **Tracking**: MLflow for experiment tracking and model versioning
- **Deployment**: AWS SageMaker hosting (endpoint: `xgb-1909`)
- **Serving**: FastAPI REST API with `/test_model` endpoint
- **Region**: EU-North-1 (Stockholm)

## Project Structure

```
SERVER_PRED/
├── aws_model.py          # FastAPI application for SageMaker inference
├── local_test.py         # Local testing script for SageMaker endpoint
├── data/
│   ├── raw/              # Raw CSV files (daily SMART data)
│   ├── processed/        # Processed training data
│   └── cheat.md          # SMART attribute reference guide
├── mlruns/               # MLflow tracking artifacts
└── YOUR_MLFLOW_TRACKING_URI/  # MLflow backend store
```

## SMART Attributes Used

The model uses key SMART attributes for failure prediction:

| ID | Attribute | Critical Threshold |
|----|-----------|-------------------|
| 1  | Raw Read Error Rate | Increasing errors |
| 5  | Reallocated Sectors | > 0 |
| 7  | Seek Error Rate | Increasing |
| 10 | Spin Retry Count | > 0 |
| 187| Reported Uncorrectable Errors | > 0 |
| 188| Command Timeout | > 0 |
| 197| Current Pending Sector | > 0 |
| 198| Uncorrectable Sector Count | > 0 |
| 199| UDMA CRC Error Count | Increasing |

## API Endpoints

### Health Check
```bash
GET /test_site
```
Response: `{"msg": "hello"}`

### Predict Failure
```bash
POST /test_model
Content-Type: application/json

{
  "data": [20, 30, 10, 20, 30, 20, 0, 0, 20, 20, 0, 0, 0]
}
```

Response:
```json
{
  "Predicted class": 0,
  "msg": ["No Failure"]
}
```

- **Input**: Array of 13 integers representing SMART attribute values
- **Output**: Predicted class (0 = No Failure, 1 = Failure) with human-readable message

## Setup

### Prerequisites

- Python 3.10+
- AWS CLI configured with SageMaker permissions
- Access to SageMaker endpoint `xgb-1909` in `eu-north-1`

### Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install fastapi uvicorn boto3 pydantic xgboost scikit-learn pandas numpy mlflow
```

### Running the API Server

```bash
uvicorn aws_model:app --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

### Testing Locally

```bash
python local_test.py
```

## Configuration

### AWS Credentials

Ensure AWS credentials are configured for the `eu-north-1` region:

```bash
aws configure
# Or set environment variables:
export AWS_ACCESS_KEY_ID=your_key
export AWS_SECRET_ACCESS_KEY=your_secret
export AWS_DEFAULT_REGION=eu-north-1
```

### SageMaker Endpoint

The endpoint name is hardcoded in `aws_model.py`:
```python
EndpointName="xgb-1909"
```

To use a different endpoint, modify the `EndpointName` parameter in the `invoke_endpoint` call.

## Model Details

- **Algorithm**: XGBoost 3.4.1
- **Task**: Binary Classification
- **Threshold**: 0.5 (probability ≥ 0.5 → Failure)
- **Features**: 13 SMART attributes
- **Training Data**: ~90 days of daily SMART logs (Jan–Mar 2020)

## Dependencies

Key packages (from MLflow artifacts):
- `mlflow==3.12.0`
- `xgboost==3.4.1`
- `scikit-learn==1.6.0`
- `pandas==2.3.3`
- `numpy==2.2.6`
- `fastapi` (for API serving)
- `boto3` (AWS SDK)

## MLflow Tracking

Experiments are tracked locally in `mlruns/` directory. To view the UI:

```bash
mlflow ui --backend-store-uri file:./mlruns
```

Then open `http://localhost:5000`

## Data Format

Input CSV files contain daily SMART attribute readings with columns corresponding to attribute IDs. The processed training data (`data/processed/train_bz_new.csv`) is used for model training.

## Security Notes

- Ensure proper IAM roles for SageMaker invocation
- Consider using AWS Secrets Manager for production deployments

## Future Improvements

- [ ] Set up CI/CD pipeline
- [ ] Add batch prediction endpoint
- [ ] Implement A/B testing for model versions

## License

Internal project - not for public distribution.
