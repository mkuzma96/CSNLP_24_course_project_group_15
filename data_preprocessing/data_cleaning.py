
import ast
import numpy as np
import pandas as pd


#%% Merge with new extract

data1 = pd.read_csv("data_initial_extraction.csv")
data2 = pd.read_csv("data_initial_bb.csv")

text_extraction = []
for response in data2['aws_responses']:
    resp = ast.literal_eval(response)
    text_extract = ''
    for item in resp['Blocks']:
        if item["BlockType"] == "WORD":
            text_extract = text_extract + ' ' + item["Text"]
    text_extraction.append(text_extract)
data2 = data2.assign(text_extract=text_extraction)
data_full = pd.merge(data2.drop("aws_responses", axis=1), data1.drop("text_extract", axis=1), on="file_name")

#%% Clean the data

# Supplier name
true_vec = np.empty(data_full.shape[0])
for i in range(data_full.shape[0]):
    if pd.isnull(data_full['supp_name'][i]):
        true_vec[i] = True
    else:
        true_vec[i] = str(data_full['supp_name'][i]) in data_full['text_extract'][i] 
data_full = data_full.iloc[np.where(true_vec == 1)[0],]
data_full.reset_index(drop=True, inplace=True)
    
# Supplier VAT
true_vec = np.empty(data_full.shape[0])
for i in range(data_full.shape[0]):
    if pd.isnull(data_full['supp_vat'][i]):
        true_vec[i] = True
    else:
        true_vec[i] = str(data_full['supp_vat'][i]) in data_full['text_extract'][i] 
data_full = data_full.iloc[np.where(true_vec == 1)[0],]
data_full.reset_index(drop=True, inplace=True)

# Date 
true_vec = np.empty(data_full.shape[0])
for i in range(data_full.shape[0]):
    if pd.isnull(data_full['date'][i]):
        true_vec[i] = True
    else:
        true_vec[i] = str(data_full['date'][i]) in data_full['text_extract'][i] 
        if true_vec[i] == False:   
            true_vec[i] = str(data_full['date'][i])[:-4] in data_full['text_extract'][i] 
data_full = data_full.iloc[np.where(true_vec == 1)[0],]
data_full.reset_index(drop=True, inplace=True)

# Amount
true_vec = np.empty(data_full.shape[0])
for i in range(data_full.shape[0]):
    if pd.isnull(data_full['amount'][i]):
        true_vec[i] = True
    else:
        true_vec[i] = str(data_full['amount'][i]) in data_full['text_extract'][i] 
data_full = data_full.iloc[np.where(true_vec == 1)[0],]
data_full.reset_index(drop=True, inplace=True)

# data_full.to_csv("data_processed_cleaned_full.csv", index=False)
