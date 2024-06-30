
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

#%% Test2 - non-seen structure

data = pd.read_csv("data_processed_cleaned_full.csv")

test_comps = ['Aldi', 'Ayverdis', 'BP', 'Bauhaus', 'Der Frisch Fish', 'Flughafen',
              'Future', 'Garage Wehntaler', 'Hotz', 'Manor', 'MediaMarkt', 'OfficeWorld',
              'Rio', 'Sense Bar', 'Volg']

test_idx = np.zeros(1512)
for comp in test_comps:
    test_idx2 = np.empty(1512)
    for i in range(1512):
        test_idx2[i] = comp in data['file_name'].values[i]
    test_idx = test_idx + test_idx2
            
data_test2 = data.iloc[np.where(test_idx == 1)[0],]            
# data_test2.to_csv("data_test2.csv", index = False)

data_new = data.iloc[np.where(test_idx == 0)[0],]   
train, test = train_test_split(data_new, test_size=0.1, random_state=42)
# train.to_csv("data_train.csv", index=False)
# test.to_csv("data_test1.csv", index=False)
