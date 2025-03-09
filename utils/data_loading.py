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

    return torch.tensor(output)

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
        self.user_reviews[row['user_id']].append(row['text'])
        self.item_reviews[row['parent_asin']].append(row['text'])

  def __len__(self):
    return len(self.df)

  def __getitem__(self, idx):
    row = self.df.iloc[idx]
    user = row['user_id']
    item = row['parent_asin']
    current_review = process_text(row['text'])
    rating = row['rating']

    """
    if the embeddings weren't pre-grouped, we should do something like this:
    user_embs = self.df[self.df['user_id'] == user]['embedding'].tolist()
    item_embs = self.df[self.df['parent_asin'] == item]['embedding'].tolist()

    this means, everytime a batch is loaded, this group by will happen, that is an overhead.
    """
    # Pre-grouped embeddings - what if user has same review text for different products? or different users give same review?
    # user_embs = [emb for emb in self.user_reviews[user] if not torch.equal(emb, current_review)]
    # item_embs = [emb for emb in self.item_reviews[item] if not torch.equal(emb, current_review)]
    
    user_embs = [process_text(review) for review in self.user_reviews[user][:self.user_reviews_per_entity]]
    item_embs = [process_text(review) for review in self.item_reviews[item][:self.item_reviews_per_entity]]

    user_embs += [torch.zeros_like(current_review)] * (self.user_reviews_per_entity - len(user_embs))
    item_embs += [torch.zeros_like(current_review)] * (self.item_reviews_per_entity - len(item_embs))

    user_tower_input = torch.stack(user_embs)
    item_tower_input = torch.stack(item_embs)

    return user_tower_input, item_tower_input, torch.tensor(rating, dtype=torch.float), user, item

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
        user_training_reviews = [process_text(review) for review in self.train_dataset.user_reviews[row['user_id']][:self.user_reviews_per_entity]]
        item_training_reviews = [process_text(review) for review in self.train_dataset.item_reviews[row['parent_asin']][:self.item_reviews_per_entity]]
        
        user_training_reviews += [torch.zeros_like(user_training_reviews[0])] \
                                    * (self.user_reviews_per_entity - len(user_training_reviews))
        
        item_training_reviews += [torch.zeros_like(item_training_reviews[0])] \
                                    * (self.item_reviews_per_entity - len(item_training_reviews))
        
        user_tower_input = torch.stack(user_training_reviews)
        item_tower_input = torch.stack(item_training_reviews)
        
        rating = row['rating']

        return user_tower_input, item_tower_input, torch.tensor(rating, dtype=torch.float), user, item


model = "NRCMA"
batch_size = 16

embedding_dim = 300
words_per_sentence = 12
sentences_per_review = 4
user_reviews_per_entity = 3
item_reviews_per_entity = 35

embed_model = downloader.load("word2vec-google-news-300")
embed_model['<unk>'] = np.random.randn(embedding_dim)

train_df = load_data('train_df_with_text.csv')
val_df = load_data('val_df_with_text.csv')
test_df = load_data('test_df_with_text.csv')

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

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)


start = time.time()

for i, batch in enumerate(train_dataloader):
    user_tower_input, item_tower_input, rating, user, item = batch
    
    if i == 10:
        break

end = time.time()
print(f"Time taken for embedding - {(end-start)/60} mins")