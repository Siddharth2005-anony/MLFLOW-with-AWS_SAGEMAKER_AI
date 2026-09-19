from sagemaker.serve.model_builder import ModelBuilder

role = "arn:aws:iam::877231425975:role/service-role/AmazonSageMaker-ExecutionRole-20260624T160238"

builder = ModelBuilder(
    image_uri="662702820516.dkr.ecr.eu-north-1.amazonaws.com/sagemaker-xgboost:3.0-5",
    s3_model_data_url="s3://sagemaker-eu-north-1-877231425975/xgb-model/model.tar.gz",
    role_arn=role,
)

# FIRST build
sm_model = builder.build(
    model_name="xgb-native-model"
)

print("Model built successfully!")

# THEN deploy
predictor = builder.deploy(
    endpoint_name="xgb-native-endpoint",
    instance_type="ml.m5.large",
    initial_instance_count=1,
)

print("Endpoint:", predictor.endpoint_name)