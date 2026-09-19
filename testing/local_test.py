import boto3

runtime = boto3.client(
    "sagemaker-runtime",
    region_name="eu-north-1"
)

test_values_1 = [
    20,  
    30,   
    10,   
    20,   
    30,   
    20,   
    0,   
    0,   
    20,  
    20,  
    0,   
    0,   
    0    
]

test_values = [
    100,  
    100,  
    100,  
    100,  
    100,  
    100,  
    100,   
    100,   
    100,   
    100,   
    100,   
    100,   
    100    
]

Body="100,100,100,100,100,100,100,100,100,100,100,100,100"

payload = ",".join(map(str, test_values))

response = runtime.invoke_endpoint(
    EndpointName="xgb-1709",
    ContentType="text/csv",
    Body=payload
)



prediction = float(
    response["Body"].read().decode()
)

label = int(prediction >= 0.5)

print("Probability:", prediction)
print("Predicted class:", label)