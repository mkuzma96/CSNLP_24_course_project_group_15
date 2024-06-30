
# Extraction with bounding boxes for AWS
import boto3
import pandas as pd

client_aws = boto3.client('textract', region_name='eu-central-1') 
s3 = boto3.resource('s3')
bucket_name = "eth-project"

my_bucket = s3.Bucket(bucket_name)
files = []
for obj in my_bucket.objects.filter(Prefix=''):
    files.append(obj.key)

file_names = []
aws_text_extractions = []
for i, file_name in zip(range(0,2014),files):
    print(i)
    response_sync = client_aws.detect_document_text(
        Document={'S3Object': {'Bucket': bucket_name, 'Name': file_name}})
    aws_text_extractions.append(response_sync)
    file_name_clean = file_name[:-4]
    file_names.append(file_name_clean)
    
data_dict = {
    'file_name': file_names,
    'aws_responses': aws_text_extractions,
    }
df_pred = pd.DataFrame(data=data_dict)
df_pred.to_csv('data_initial_bb.csv', index=False)
