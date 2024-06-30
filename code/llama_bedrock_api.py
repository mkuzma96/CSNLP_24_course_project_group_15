
# Performance of Llama models
import pandas as pd
import boto3
import json
import pickle
client = boto3.client("bedrock-runtime", region_name="us-east-1")

model_name = "llama3-8b" # Change to "llama3-70b" for the other result
model_name_full = "meta." + model_name + "-instruct-v1:0"
    
def get_response_llama(prompt, model=model_name_full):
    formatted_prompt = f"""
    <|begin_of_text|>
    <|start_header_id|>user<|end_header_id|>
    {prompt}
    <|eot_id|>
    <|start_header_id|>assistant<|end_header_id|>
    """
    native_request = {
        "prompt": formatted_prompt,
        "max_gen_len": 512,
        "temperature": 0,
    }
    request = json.dumps(native_request)
    response = client.invoke_model(modelId=model_name_full, body=request)
    model_response = json.loads(response["body"].read())
    return model_response["generation"]

#%% Extraction output

d_test_name = "test1" # Change to "test2" for the other result
d_test_name_full = "data_" + d_test_name + ".csv"
data_test = pd.read_csv(d_test_name_full)
k = 1 # Change to 3 and 5 for the other results
prompt_file = "prompt_" + str(k) + "-shot.txt"
with open(prompt_file, 'r') as file:
    prompt_text = file.read()
information_extractions = []
for i in range(data_test.shape[0]):
    
    # Start extraction pipeline with a prompt
    text_extract = data_test['text_extract'].values[i]
    prompt = prompt_text.format(text_extract=text_extract)
      
    # Get and store extraction results
    info_extraction = get_response_llama(prompt)
    information_extractions.append(info_extraction)
    
# Save extraction results
result_file_name = d_test_name + "_" + model_name + '_' + str(k) + '-shot' + ".pickle"
with open(result_file_name, "wb") as file:   
    pickle.dump(information_extractions, file, protocol=pickle.HIGHEST_PROTOCOL)