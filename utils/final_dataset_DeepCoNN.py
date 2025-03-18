import time
import torch
import zipfile
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader

# -------------------------------
# PyTorch Dataset Class
# -------------------------------
# Train Dataset

class DeepConnDataset(Dataset):
    """
    The input dataframe should be the original dataframe, along with an additional column where we have the preprocessed embeddings matrix
    of (sentences in review, words in a sentence, emb dimension) size.
    """
    def __init__(self, df, user_tower_inputs, item_tower_inputs):
        self.df = df
        self.user_tower_inputs = user_tower_inputs
        self.item_tower_inputs = item_tower_inputs

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        u_id = torch.tensor(row['user_id'], dtype=torch.long)
        i_id = torch.tensor(row['parent_asin'], dtype=torch.long)
        rating = torch.tensor(row['rating'], dtype=torch.float32)

        user_tower_input = self.user_tower_inputs[idx]
        item_tower_input = self.item_tower_inputs[idx]

        return u_id, i_id, user_tower_input, item_tower_input, rating
    

class DeepConnTestDataset(Dataset):
    """
    The input dataframe should be the original dataframe, along with an additional column where we have the preprocessed embeddings matrix
    of (sentences in review, words in a sentence, emb dimension) size.
    """
    def __init__(self, test_df, user_tower_inputs, item_tower_inputs):
        self.test_df = test_df
        self.user_tower_inputs = user_tower_inputs
        self.item_tower_inputs = item_tower_inputs

    def __len__(self):
        return len(self.test_df)
    
    def __getitem__(self, idx):
        row = self.test_df.iloc[idx]

        user = row['user_id']
        item = row['parent_asin']
        rating = row['rating']
        train_user_idx = row['train_user_idx']
        train_item_idx = row['train_item_idx']

        u_id = torch.tensor([user], dtype=torch.int).squeeze(-1)
        i_id = torch.tensor([item], dtype=torch.int).squeeze(-1)
        rating = torch.tensor([rating], dtype=torch.float).squeeze(-1)
        
        user_tower_input = self.user_tower_inputs[train_user_idx]
        item_tower_input = self.item_tower_inputs[train_item_idx]

        return u_id, i_id, user_tower_input, item_tower_input, rating
    

train_df = pd.read_csv('./data/train_df_filtered.csv')
val_df = pd.read_csv('./data/val_df_filtered.csv')
test_df = pd.read_csv('./data/test_df_filtered.csv')

user_tower_inputs = torch.load('./data/user_tower_input_DEEPCONN.pt').to(torch.int)
item_tower_inputs = torch.load('./data/item_tower_input_DEEPCONN.pt').to(torch.int)
train_dataset = DeepConnDataset(train_df, user_tower_inputs, item_tower_inputs)
val_dataset = DeepConnTestDataset(val_df, user_tower_inputs, item_tower_inputs)
test_dataset = DeepConnTestDataset(test_df, user_tower_inputs, item_tower_inputs)

shuffle = False
pin_memory = False
num_workers = 0

batch_size = 512

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=pin_memory, num_workers=num_workers)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=pin_memory, num_workers=num_workers)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=pin_memory, num_workers=num_workers)

start = time.time()

for i, batch in enumerate(train_dataloader):
    user_tower_input, item_tower_input, rating, user, item = batch
    
    if i == 10:
        break

end = time.time()
print(f"Time taken for embedding - {(end-start)/60} mins")