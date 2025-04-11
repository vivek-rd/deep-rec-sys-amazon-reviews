import pandas as pd
import time
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

# -------------------------------
# PyTorch Dataset Class
# -------------------------------
# Train Dataset
class NRCMARecSysDataset(Dataset):

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
    
    user = row['user_id_cat']
    item = row['parent_asin_cat']
    item_id = row['parent_asin'] # item_id will be a string
    rating = row['rating']
    
    user = torch.tensor([user], dtype=torch.int).squeeze(-1)
    item = torch.tensor([item], dtype=torch.int).squeeze(-1)
    rating = torch.tensor([rating], dtype=torch.float).squeeze(-1)
    
    user_tower_input = self.user_tower_inputs[idx]
    item_tower_input = self.item_tower_inputs[idx]

    return user_tower_input, item_tower_input, rating, user, item, item_id


class NRCMARecSysTestDataset(Dataset):

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
    
    user = row['user_id_cat']
    item = row['parent_asin_cat']
    rating = row['rating']
    train_user_idx = row['train_user_idx']
    train_item_idx = row['train_item_idx']
    
    user = torch.tensor([user], dtype=torch.int).squeeze(-1)
    item = torch.tensor([item], dtype=torch.int).squeeze(-1)
    rating = torch.tensor([rating], dtype=torch.float).squeeze(-1)
    
    user_tower_input = self.user_tower_inputs[train_user_idx]
    item_tower_input = self.item_tower_inputs[train_item_idx]

    return user_tower_input, item_tower_input, rating, user, item


train_df = pd.read_csv('data/train_df_filtered.csv')
val_df = pd.read_csv('data/val_df_filtered.csv')
test_df = pd.read_csv('data/test_df_filtered.csv')

user_tower_inputs = torch.load('data/user_tower_input_NRCMA.pt')
item_tower_inputs = torch.load('data/item_tower_input_NRCMA.pt')
train_dataset = NRCMARecSysDataset(train_df, user_tower_inputs, item_tower_inputs)
val_dataset = NRCMARecSysTestDataset(val_df, user_tower_inputs, item_tower_inputs)
test_dataset = NRCMARecSysTestDataset(test_df, user_tower_inputs, item_tower_inputs)

shuffle = True
pin_memory = True
num_workers = 0

batch_size = 512

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=pin_memory, num_workers=num_workers)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=pin_memory, num_workers=num_workers)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=pin_memory, num_workers=num_workers)

# start = time.time()

# for i, batch in enumerate(train_dataloader):
#     user_tower_input, item_tower_input, rating, user, item = batch
    
#     if i == 10:
#         break

# end = time.time()
# print(f"Time taken for embedding - {(end-start)/60} mins")