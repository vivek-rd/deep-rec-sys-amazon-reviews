import pandas as pd
import numpy as np
import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import gensim.downloader as downloader
import nltk
from tensorflow.keras.preprocessing.text import text_to_word_sequence
from nltk import word_tokenize
# from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

def get_list_dicts(file):
    return [json.loads(line) for line in open(file, "rt")]

# raw_data = get_list_dicts("/Users/rutvikdhopate/Downloads/Magazine_Subscriptions.jsonl")
raw_data = get_list_dicts("Magazine_Subscriptions.jsonl")
df = pd.DataFrame(raw_data).loc[:,['user_id','asin','rating','text']]
df = df.iloc[:500,:]
# print(df.shape)
df['text'] = df['text'].astype(str)
df['text'] = df['text'].apply(lambda x: ' '.join(text_to_word_sequence(x)))
df = df[df['text'].str.len() > 1]
df.drop_duplicates(subset=['user_id','asin'], inplace=True)
df.dropna(inplace=True)
# print(df.shape)
# print(df.head())

# Idea is to group the reviews based on the users and the items as inputs for 2 parallel Neural Nets
# grouped_u = df.groupby('user_id')['text'].apply(' <SEP> '.join).reset_index()
# grouped_i = df.groupby('asin')['text'].apply(' <SEP> '.join).reset_index()

grouped_u = df.groupby('user_id').agg(
    text=('text', ' <SEP> '.join),  # Concatenate reviews for each user
    rating=('rating', 'mean')     # Calculate the average rating for each user
).reset_index()
grouped_i = df.groupby('asin').agg(
    text=('text', ' <SEP> '.join),  # Concatenate reviews for each item
    rating=('rating', 'mean')     # Calculate the average rating for each item
).reset_index()

# Number of Unique Users and Unique Items in the dataset
print(f"Unique Users: {grouped_u.shape[0]}, Unique Items: {grouped_i.shape[0]}")

# Google Word-2-vec word embeddings - as mentioned in the paper
embeds = downloader.load('word2vec-google-news-300')

# Add 3 new tokens to the embeds dictionary
'''
    <UNK> - Representation for the word that is not a part of the pretrained embeddings
    <SEP> - The reviews are concatenated by a token to indicate that there's a separation
    <PAD> - 300 zeros to match the length of the sentence. 
'''

embeds['<UNK>'] = np.random.randn(300).astype(np.float32)
embeds['<SEP>'] = np.random.randn(300).astype(np.float32)
embeds['<PAD>'] = np.zeros(300, dtype=np.float32)

# Convert the sentences to tokens
nltk.download('punkt_tab')
grouped_u['text'] = grouped_u['text'].apply(lambda x: word_tokenize(x))
grouped_i['text'] = grouped_i['text'].apply(lambda x: word_tokenize(x))


# Represent the words as a fixed length; thus PAD or TRIM the tokens respectively
def create_embeddings(review_text, max_length, embedding_dict):
    if len(review_text) > max_length:
        review_text = review_text[:max_length]
    else:
        review_text = review_text + (max_length-len(review_text))*['<PAD>']
    
    # Now that the length is unique, map the words from the embedding_dict
    text_embeddings = np.array([embedding_dict[token] if token in embedding_dict else embedding_dict['<PAD>'] for token in review_text])
    return text_embeddings
    

# Create the Embeddings with max_length of 200 words for concatenated user reviews and 1000 words for concatenated item reviews
grouped_u['text_embeddings'] = grouped_u['text'].apply(lambda x: create_embeddings(x, max_length=200, embedding_dict=embeds))
grouped_i['text_embeddings'] = grouped_i['text'].apply(lambda x: create_embeddings(x, max_length=200, embedding_dict=embeds))

print("Checkpoint - Embeddings")
# Need to Encode the user_id and asin as well
# user_id_encoding = {user: idx for idx, user in enumerate(grouped_u['user_id'])}
# asin_encoding = {item: idx for idx, item in enumerate(grouped_i['asin'])}


# # Data Splitting - 80% train, 10% val, 10% test
# train_u, test_u = train_test_split(grouped_u, test_size=0.2, random_state=42)
# valid_u, test_u = train_test_split(test_u, test_size=0.5, random_state=42)

# # Split the grouped_i (item reviews) into train, validation, and test sets
# train_i, test_i = train_test_split(grouped_i, test_size=0.2, random_state=42)
# valid_i, test_i = train_test_split(test_i, test_size=0.5, random_state=42)

user_x = torch.tensor(np.stack(grouped_u['text_embeddings'].values), dtype=torch.float)
user_y = torch.tensor(grouped_u['rating'], dtype=torch.float)
item_x = torch.tensor(np.stack(grouped_i['text_embeddings'].values), dtype=torch.float)
item_y = torch.tensor(grouped_i['rating'], dtype=torch.float)


# Preparing the embedding weight tensor
embedding_dim = embeds.vector_size
vocab_size = len(embeds.key_to_index)

wtoi = {word: idx for idx, word in enumerate(embeds.key_to_index.keys())}
embedding_matrix = np.zeros((vocab_size, embedding_dim))

# Fill the matrix with pre trained embeddings
for word, idx in wtoi.items():
    try:
        # Try to get the vector for the word
        vector = embeds[word]
        embedding_matrix[idx] = vector
    except KeyError:
        # If the word is not found, initialize with a random vector
        embedding_matrix[idx] = np.random.randn(embedding_dim)

# Create a tensor from the embedding matrix
embedding_weight = torch.tensor(embedding_matrix, dtype=torch.float)


print("Checkpoint - Embeddings 2")
# Trial at the DeepCoNN Neural Network Architecture in Python
# Hyperparameters
max_review_length_u = 200
max_review_length_i = 1000
embed_dim = 300
t = [3, 5]                  # Kernel Width
n1 = 100                    # Kernel Depth
latent_factors = 50 
fm_k = 10           # Number of factors in Factorization Machine
reg_lambda = 2e-3   # Regularization Lambda
batch_size = 100
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Convolution Max-Pooling Layer
class ConvMaxLayer(torch.nn.Module):
    '''
        The independent layer for user review and item review
    '''
    def __init__(self, max_review_length, t, embed_dim, n1, latent_factors):
        super().__init__()
        self.max_review_length = max_review_length
        # self.max_review_length_u = max_review_length_u
        # self.max_review_length_i = max_review_length_i
        self.t = t
        self.embed_dim = embed_dim
        self.n1 = n1
        self.latent_factors = latent_factors

        self.convs = torch.nn.ModuleList()
        self.maxs = torch.nn.ModuleList()

        # self.convs_u = torch.nn.ModuleList()
        # self.maxs_u = torch.nn.ModuleList()

        # self.convs_i = torch.nn.ModuleList()
        # self.maxs_i = torch.nn.ModuleList()

        for width in t:
            self.convs.append(
                torch.nn.Conv1d(
                    in_channels = embed_dim,
                    out_channels = n1,
                    kernel_size = width,
                    stride=1
                )
            )
            self.maxs.append(
                torch.nn.MaxPool1d(
                    kernel_size = self.max_review_length - width + 1,
                    stride=1
                )
            )
        # for width in t:
        #     self.convs_u.append(
        #         torch.nn.Conv1d(
        #             in_channels=embed_dim,
        #             out_channels=n1,
        #             kernel_size=width,
        #             stride=1
        #         )
        #     )
        #     self.maxs_u.append(
        #         torch.nn.MaxPool1d(
        #             kernel_size=self.max_review_length_u - width + 1,
        #             stride=1
        #         )
        #     )

        #     self.convs_i.append(
        #         torch.nn.Conv1d(
        #             in_channels=embed_dim,
        #             out_channels=n1,
        #             kernel_size=width,
        #             stride=1
        #         )
        #     )
        #     self.maxs_i.append(
        #         torch.nn.MaxPool1d(
        #             kernel_size=self.max_review_length_i - width + 1,
        #             stride=1
        #         )
        #     )
        
        self.activation = torch.nn.ReLU()       # Shared activation function for user and item
        self.full_connect = torch.nn.Linear(self.n1 * len(self.t), self.latent_factors)     # Shared fully connected layer

    
    def forward(self, review):
        """
            Input Shape: (Batch Size, Review Length, Word Embedding Size)
            Output Shape: (Batch Size, Latent Factors Size)
        """
        # output_u = []
        # output_i = []
        # review_u = review_u.permute(0,2,1)
        # for max_pool, conv in zip(self.maxs_u, self.convs_u):
        #     out = self.activation(conv(review_u))
        #     max_out = max_pool(out)
        #     flatten_out = torch.flatten(max_out, start_dim=1)
        #     output_u.append(flatten_out)

        # for max_pool, conv in zip(self.maxs_i, self.convs_i):
        #     out = self.activation(conv(review_i))
        #     max_out = max_pool(out)
        #     flatten_out = torch.flatten(max_out, start_dim=1)
        #     output_i.append(flatten_out)

        # conv_out_u = torch.cat(output_u, dim=1)
        # conv_out_i = torch.cat(output_i, dim=1)

        # conv_out = torch.cat([conv_out_u, conv_out_i], dim=1)
        # latent = self.full_connect(conv_out)

        output = []
        review = review.permute(0,2,1)
        for max_pool, conv in zip(self.maxs, self.convs):
            out = self.activation(conv(review))
            max_out = max_pool(out)
            flatten_out = torch.flatten(max_out, stride=1)
            output.append(flatten_out)
        
        conv_out = torch.cat(output, dim=1)
        latent = self.full_connect(conv_out)

        return latent
    

class FMLayer(torch.nn.Module):
    """
        Factorization Machine
        Reference: https://www.kaggle.com/gennadylaptev/factorization-machine-implemented-in-pytorch
        Input Shape: (Batch Size, Latent Factors Size * 2)
        Output Shape: (Batch Size)
    """

    def __init__(self, latent_factors, fm_k):
        super().__init__()
        self.latent_factors = latent_factors
        self.fm_k = fm_k
        self.V = torch.nn.Parameter(torch.randn(self.latent_factors * 2, self.fm_k))
        self.lin = torch.nn.Linear(self.latent_factors * 2, 1)

    def forward(self, x):
        s1_square = torch.matmul(x, self.V).pow(2).sum(1, keepdim=True)
        s2 = torch.matmul(x.pow(2), self.V.pow(2)).sum(1, keepdim=True)

        out_inter = 0.5 * (s1_square-s2)
        out_lin = self.lin(out_inter)
        out = out_inter + out_lin
        return out
    

class DeepCoNN(nn.Module):
    def __init__(self, embedding_weight, max_review_length_u, max_review_length_i, t, embed_dim, n1, latent_factors):
        super().__init__()

        self.embedding_weight = embedding_weight
        self.max_review_length_u = max_review_length_u
        self.max_review_length_i = max_review_length_i
        self.t = t
        self.embed_dim = embed_dim
        self.n1 = n1
        self.latent_factors = latent_factors


        self.embedding = torch.nn.Embedding.from_pretrained(embedding_weight)
        self.embedding.weight.requires_grad = False

        self.user_layer = ConvMaxLayer(max_review_length_u, t, embed_dim, n1, latent_factors)
        self.item_layer = ConvMaxLayer(max_review_length_i, t, embed_dim, n1, latent_factors)
        self.share_layer = FMLayer()

    def forward(self, user_review, item_review):
        """
            Input Shape: (Batch Size, Review Length)
            Output ShapeL (Batch Size)
        """

        user_review = self.embedding(user_review)
        item_review = self.embedding(item_review)
        user_latent = self.user_layer(user_review)
        item_latent = self.item_layer(item_review)
        latent = torch.cat([user_latent, item_latent], dim=1)
        predict = self.share_layer(latent)
        return predict

print("Checkpoint - Architecture Done")

# An attempt at a basic training loop
user_x, user_y, item_x, item_y = user_x.to(device), user_y.to(device), item_x.to(device), item_y.to(device)
print(user_x.shape, user_y.shape, item_x.shape, item_y.shape)
train_data = TensorDataset(user_x, item_x, user_y, item_y)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)


model = DeepCoNN(embedding_weight, max_review_length_u, max_review_length_i, t, embed_dim, n1, latent_factors).to(device)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

print("Checkpoint - Training Loop Reached")
num_epochs = 3
for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for user_batch, item_batch, user_labels, item_labels in train_loader:
        optimizer.zero_grad()
        predictions = model(user_batch, item_batch)
        loss = criterion(predictions, user_labels)
        loss.backward()
        optimizer.step()
        total_loss+=loss.item()
    
    avg_loss = total_loss/len(train_loader)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
