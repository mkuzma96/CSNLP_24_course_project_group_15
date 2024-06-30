
# Finetuning Mistral models
import os
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
import pandas as pd
import pickle

api_key = os.environ.get("MISTRAL_API_KEY")
client = MistralClient(api_key=api_key)
model_name = 'mistral-7b-finetuned'

def get_response_mistral_finetune(prompt, retrieved_job):
    chat_response = client.chat(
        model=retrieved_job.fine_tuned_model,
        messages=[ChatMessage(role='user', content=prompt)]
    )
    return chat_response.choices[0].message.content

#%% Extraction output

d_test_name = "test1" # Change to "test2" for the other result
d_test_name_full = "data_" + d_test_name + ".csv"
data_test = pd.read_csv(d_test_name_full)
k = 1 # Change to 3 and 5 for the other results
if k == 1:
    model_id = 'a57c398c-5f11-431e-8fe4-a706df3801b1'
elif k == 3:
    model_id = '00c7d315-b47b-4740-b499-98e904acb1ae'
elif k == 5:
    model_id = '842df0fd-74e1-4a11-88d8-1c3965d2852d'
retrieved_job = client.jobs.retrieve(model_id)
prompt_file = "prompt_" + str(k) + "-shot.txt"
with open(prompt_file, 'r') as file:
    prompt_text = file.read()
information_extractions = []
for i in range(data_test.shape[0]):
    
    # Start extraction pipeline with a prompt
    text_extract = data_test['text_extract'].values[i]
    prompt = prompt_text.format(text_extract=text_extract)
        
    # Get and store extraction results
    info_extraction = get_response_mistral_finetune(prompt, retrieved_job)
    information_extractions.append(info_extraction)
    
# Save extraction results
result_file_name = d_test_name + "_" + model_name + '_' + str(k) + '-shot' + ".pickle"
with open(result_file_name, "wb") as file:   
    pickle.dump(information_extractions, file, protocol=pickle.HIGHEST_PROTOCOL)