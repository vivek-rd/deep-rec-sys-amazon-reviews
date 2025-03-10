import yaml
import time
import torch
import nltk
import pandas as pd
import numpy as np
from collections import defaultdict

import gensim.downloader as downloader
from torch.utils.data import Dataset, DataLoader

def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

def process_text(text, model='NRCMA'):
    
    if model == 'HSACN':
        output = np.zeros((sentences_per_review, words_per_sentence, embedding_dim))
        sentences = nltk.sent_tokenize(str(text))

        for i, sentence in enumerate(sentences[0:sentences_per_review]):
            words = nltk.word_tokenize(sentence)
            for j, word in enumerate(words[0:words_per_sentence]):
                if word in embed_model:
                    output[i, j] = embed_model[word]
                else:
                    output[i, j] = embed_model['<unk>']
    
    elif model == 'NRCMA':
        output = np.zeros((words_per_sentence, embedding_dim))
        words = nltk.word_tokenize(str(text))

        for i, word in enumerate(words[0:words_per_sentence]):
            if word in embed_model:
                output[i, :] = embed_model[word]
            else:
                output[i, :] = embed_model['<unk>']

    return torch.from_numpy(output).to(torch.float32)

# Train Dataset
class NRCMARecSysDataset(Dataset):

  """
  The input dataframe should be the original dataframe, along with an additional column where we have the preprocessed embeddings matrix
  of (sentences in review, words in a sentence, emb dimension) size.
  """

  def __init__(self, df, user_reviews_per_entity, item_reviews_per_entity):
    self.df = df
    self.user_reviews_per_entity = user_reviews_per_entity
    self.item_reviews_per_entity = item_reviews_per_entity

    # Pre-grouping user and item reviews instead of fetching it everytime in getitem
    self.user_reviews = defaultdict(list)
    self.item_reviews = defaultdict(list)

    for _, row in df.iterrows():
        self.user_reviews[row['user_id']].append((row['user_id'], row['text']))
        self.item_reviews[row['parent_asin']].append((row['user_id'], row['text']))

  def __len__(self):
    return len(self.df)

  def __getitem__(self, idx):
    row = self.df.iloc[idx]
    user = row['user_id']
    item = row['parent_asin']
    current_review = row['text']
    rating = row['rating']

    """
    if the embeddings weren't pre-grouped, we should do something like this:
    user_embs = self.df[self.df['user_id'] == user]['embedding'].tolist()
    item_embs = self.df[self.df['parent_asin'] == item]['embedding'].tolist()

    this means, everytime a batch is loaded, this group by will happen, that is an overhead.
    """
    
    user_embs = [process_text(review[1]) for review in self.user_reviews[user][:self.user_reviews_per_entity] \
                                        if current_review != review[1] and review[0] != user]
    item_embs = [process_text(review[1]) for review in self.item_reviews[item][:self.item_reviews_per_entity] \
                                        if current_review != review[1] and review[0] != item]

    user_embs += [torch.zeros(words_per_sentence, embedding_dim)] * (self.user_reviews_per_entity - len(user_embs))
    item_embs += [torch.zeros(words_per_sentence, embedding_dim)] * (self.item_reviews_per_entity - len(item_embs))

    user_tower_input = torch.stack(user_embs).to(torch.float32)
    item_tower_input = torch.stack(item_embs).to(torch.float32)
    
    user = torch.tensor([user], dtype=torch.int).squeeze(-1)
    item = torch.tensor([item], dtype=torch.int).squeeze(-1)
    rating = torch.tensor([rating], dtype=torch.float).squeeze(-1)

    return user_tower_input, item_tower_input, rating, user, item

# Validation and test Dataset
class NRCMARecSysTestDataset(Dataset):

    def __init__(self, df, train_dataset, user_reviews_per_entity, item_reviews_per_entity):
        self.df = df
        self.train_dataset = train_dataset
        self.user_reviews_per_entity = user_reviews_per_entity
        self.item_reviews_per_entity = item_reviews_per_entity

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        user = row['user_id']
        item = row['parent_asin']
        user_training_reviews = [process_text(review[1]) for review in self.train_dataset.user_reviews[row['user_id']][:self.user_reviews_per_entity]]
        item_training_reviews = [process_text(review[1]) for review in self.train_dataset.item_reviews[row['parent_asin']][:self.item_reviews_per_entity]]
        
        user_training_reviews += [torch.zeros(words_per_sentence, embedding_dim)] \
                                    * (self.user_reviews_per_entity - len(user_training_reviews))
        
        item_training_reviews += [torch.zeros(words_per_sentence, embedding_dim)] \
                                    * (self.item_reviews_per_entity - len(item_training_reviews))
        
        user_tower_input = torch.stack(user_training_reviews).to(torch.float32)
        item_tower_input = torch.stack(item_training_reviews).to(torch.float32)
        
        user = torch.tensor([user], dtype=torch.int).squeeze(-1)
        item = torch.tensor([item], dtype=torch.int).squeeze(-1)
        rating = torch.tensor([row['rating']], dtype=torch.float).squeeze(-1)

        return user_tower_input, item_tower_input, rating, user, item

model = "NRCMA"

# load the current config file
with open('config/nrcma.yaml') as f:
    config = yaml.safe_load(f)

batch_size = config['t']['batch_size']
words_per_sentence = config['m']['words_per_sentence']
sentences_per_review = 4
user_reviews_per_entity = config['m']['user_reviews_per_entity']
item_reviews_per_entity = config['m']['item_reviews_per_entity']

embedding_dim = 300
embed_model = downloader.load("word2vec-google-news-300")
embed_model.add_vector('<unk>', np.random.randn(embedding_dim))

train_df = load_data('/Users/vivekrachakonda/Documents/Courses/Capstone/github/deep-rec-sys-amazon-reviews/utils/train_df_filtered.csv')
val_df = load_data('/Users/vivekrachakonda/Documents/Courses/Capstone/github/deep-rec-sys-amazon-reviews/utils/val_df_filtered.csv')
test_df = load_data('/Users/vivekrachakonda/Documents/Courses/Capstone/github/deep-rec-sys-amazon-reviews/utils/test_df_filtered.csv')

user_ids = set(train_df['user_id'].dropna().unique())
item_ids = set(train_df['parent_asin'].dropna().unique())

user_codes = pd.CategoricalDtype(user_ids)
item_codes = pd.CategoricalDtype(item_ids)

for df in [train_df, val_df, test_df]:
    df['user_id'] = df['user_id'].astype(user_codes).cat.codes
    df['parent_asin'] = df['parent_asin'].astype(item_codes).cat.codes
    

train_dataset = NRCMARecSysDataset(train_df, user_reviews_per_entity, item_reviews_per_entity)
val_dataset = NRCMARecSysTestDataset(val_df, train_dataset, user_reviews_per_entity, item_reviews_per_entity)
test_dataset = NRCMARecSysTestDataset(test_df, train_dataset, user_reviews_per_entity, item_reviews_per_entity)

shuffle = False
pin_memory = False
num_workers = 0

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=pin_memory, num_workers=num_workers)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=pin_memory, num_workers=num_workers)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=pin_memory, num_workers=num_workers)


# start = time.time()

# for i, batch in enumerate(train_dataloader):
#     user_tower_input, item_tower_input, rating, user, item = batch
    
#     if i == 100:
#         break

# end = time.time()
# print(f"Time taken for embedding - {(end-start)/60} mins")