
import boto3
import pickle
import pandas as pd
import time

client_aws = boto3.client('textract', region_name='eu-central-1') 
bucket = "eth-project"

#%% Extraction output

d_test_name = "test1" # Change to "test2" for the other result
d_test_name_full = "data_" + d_test_name + ".csv"
data_test = pd.read_csv(d_test_name_full)
information_extractions = []
for i in range(data_test.shape[0]):
    
    # Start extraction pipeline
    file_name = data_test['file_name'].values[i] + ".pdf"
    job_id = client_aws.start_expense_analysis(
        DocumentLocation={'S3Object': {'Bucket': bucket, 'Name': file_name}})['JobId']
    wait_time = 5
    time.sleep(wait_time)
    response = client_aws.get_expense_analysis(
        JobId=job_id)
    status = response["JobStatus"]
    while status == "IN_PROGRESS":
        time.sleep(wait_time)
        response = client_aws.get_expense_analysis(
            JobId=job_id)
        status = response["JobStatus"]
        
    # Get extracted entities
    AWS_extract = {
            'Supplier company name': [], 
            'Supplier VAT number': [], 
            'Date': [], 
            'Amount': [] 
        }
    for exp_ind in response["ExpenseDocuments"]:
        for field in exp_ind['SummaryFields']:
            if field['Type']['Text'] == 'VENDOR_NAME':
                AWS_extract['Supplier company name'] = field['ValueDetection']['Text']    
            if field['Type']['Text'] == 'VENDOR_VAT_NUMBER':
                AWS_extract['Supplier VAT number'] = field['ValueDetection']['Text']    
            if field['Type']['Text'] == 'INVOICE_RECEIPT_DATE':
                AWS_extract['Date'] = field['ValueDetection']['Text']
            if field['Type']['Text'] == 'TOTAL':
                 AWS_extract['Amount'] = field['ValueDetection']['Text']  
                 
    # Store extraction results
    information_extractions.append(AWS_extract)

# Save extraction results
result_file_name = d_test_name + "_aws.pickle"
with open(result_file_name, "wb") as file:   
    pickle.dump(information_extractions, file, protocol=pickle.HIGHEST_PROTOCOL)
    
    