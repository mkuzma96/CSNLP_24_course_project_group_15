
# Performance of GPT models
import os
import pandas as pd
from openai import OpenAI
import pickle

client_openai = OpenAI(
    api_key= os.getenv("OPENAI_API_KEY")
)

model_name = "gpt-3.5" # Change to "gpt-4" for the other result
model_name_full = model_name + "-turbo"
    
def get_response_gpt(prompt, model=model_name_full):
    messages = [{"role": "user", "content": prompt}]
    response = client_openai.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0 
    )
    return response.choices[0].message.content


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
    info_extraction = get_response_gpt(prompt)
    information_extractions.append(info_extraction)
    
# Save extraction results
result_file_name = d_test_name + "_" + model_name + '_' + str(k) + '-shot' + ".pickle"
with open(result_file_name, "wb") as file:   
    pickle.dump(information_extractions, file, protocol=pickle.HIGHEST_PROTOCOL)
    