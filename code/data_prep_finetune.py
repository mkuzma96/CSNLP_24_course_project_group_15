import pandas as pd
import json

# Load the CSV file
data = pd.read_csv("data_train.csv")
k = 1 # Change to 3 and 5 for the other results

# Define a function to convert each row into instruction format
def create_instruction(row):
    
    # Prompt
    prompt_file = "prompt_" + str(k) + "-shot.txt"
    with open(prompt_file, 'r') as file:
        prompt_text = file.read()
    prompt = prompt_text.format(text_extract=row['text_extract']) 
    
    # Response
    response = json.dumps({
        "Supplier company name": row["supp_name"],
        "Supplier VAT number": row["supp_vat"],
        "Date": row["date"],
        "Amount": row["amount"]
    })
    
    # Generate line in Mistral fine-tuning format
    finetune_line = {
      "messages": [
        {
          "role": "user",
          "content": f"{prompt}"
        },
        {
          "role": "assistant",
          "content": f"{response}"
        }
      ]
    }
    return finetune_line

# Apply the function to each row and convert to list of dictionaries
instruction_dataset = data.apply(create_instruction, axis=1).tolist()

# Save the dataset to a JSON file
instruct_data_file = "instruct_data_" + str(k) + "-shot.jsonl"
with open(instruct_data_file, "w") as file:
    for item in instruction_dataset:
        file.write(json.dumps(item) + '\n')
