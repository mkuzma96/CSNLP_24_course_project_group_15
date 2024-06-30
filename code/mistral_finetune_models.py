
# Finetuning Mistral models
import os
from mistralai.client import MistralClient
from mistralai.models.jobs import TrainingParameters
from mistralai.models.chat_completion import ChatMessage

api_key = os.environ.get("MISTRAL_API_KEY")
client = MistralClient(api_key=api_key)

#%% Set up finetuning jobs

k = 1 # Change to 3 and 5 for the other results
instruct_data_file = "instruct_data_" + str(k) + "-shot.jsonl"
with open(instruct_data_file, "rb") as f:
    instruct_data_finetune = client.files.create(file=(instruct_data_file, f))

client.jobs.create(
    model="open-mistral-7b",
    training_files=[instruct_data_finetune.id],
    hyperparameters=TrainingParameters(
        training_steps=100,
        learning_rate=0.0001,
        )
)

jobs = client.jobs.list()
print(jobs)

#%% Retrieve and test finetuned model

job_id = jobs.data[0].id
retrieved_job = client.jobs.retrieve(job_id)
chat_response = client.chat(
    model=retrieved_job.fine_tuned_model,
    messages=[ChatMessage(role='user', content="How are you doing?")]
)
print(chat_response.choices[0].message.content)
