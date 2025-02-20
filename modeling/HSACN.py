# %% [markdown]
# <a href="https://colab.research.google.com/github/vivek-rd/deep-rec-sys-amazon-reviews/blob/dev/modeling/HSACN.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# %%

# %%
import numpy as np
import pandas as pd

# %%
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

# %%
import re
import math
import seaborn as sns
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from datasets import load_dataset
import gensim.downloader as downloader

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

# %%
#this is basically users dataset coz only users have the associated reviews, not the items. So we don't need items data at all if only going with reviews.

dataset = load_dataset("McAuley-Lab/Amazon-Reviews-2023", "raw_review_Appliances", split= 'full', trust_remote_code=True)
dataset[0]

# %%
sample_dataset= dataset.take(1000)

# %% [markdown]
# # Analysis

# %%
df= sample_dataset.to_pandas()
df.head()

# %%
df = pd.read_csv("../data/books_mini_sample.csv")
print(len(df))
df.head()

# %%
df['word_count'] = df['text'].apply(lambda x: len(x.split()))
df['sentence_count'] = df['text'].apply(lambda x: len(sent_tokenize(x)))

print("Word Count Statistics:")
print(df['word_count'].describe())

print("\nSentence Count Statistics:")
print(df['sentence_count'].describe())

# %%
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.hist(df['word_count'], bins=50, color='skyblue', edgecolor='black')
plt.title("Distribution of Word Counts per Review")
plt.xlabel("Word Count")
plt.ylabel("Frequency")

plt.subplot(1, 2, 2)
plt.hist(df['sentence_count'], bins=50, color='salmon', edgecolor='black')
plt.title("Distribution of Sentence Counts per Review")
plt.xlabel("Sentence Count")
plt.ylabel("Frequency")

plt.tight_layout()
plt.show()

# %% [markdown]
# Most of the Word Count is concentrated between 0 to 500 and Sentence Count between 0 to 30 atmost, so zooming in on that.

# %%
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.hist(df['word_count'], bins=50, color='skyblue', edgecolor='black')
plt.title("Distribution of Word Counts per Review")
plt.xlabel("Word Count")
plt.ylabel("Frequency")
plt.xlim(0, 80) #plt.xlim(0, 500)

plt.subplot(1, 2, 2)
plt.hist(df['sentence_count'], bins=50, color='salmon', edgecolor='black')
plt.title("Distribution of Sentence Counts per Review")
plt.xlabel("Sentence Count")
plt.ylabel("Frequency")
plt.xlim(0, 10) #plt.xlim(0, 30)

plt.tight_layout()
plt.show()

# %% [markdown]
# So it's safe to assume we would have 100 words per review, and 7 sentences per review (after changing the limits, the number changed to 70, 4 on further experiments). The catch is, we want words per sentences for the sentence encoder and not words per review. It would be time consuming to calculate that so we can settle for approximating by (words_per_review / sentence_per_review) to get words_per_sentence.

# %%
avg_words = df['word_count'].mean()
avg_sentences = df['sentence_count'].mean()

avg_words_per_sentence = avg_words / avg_sentences

print("Average words per review:", avg_words)
print("Average sentences per review:", avg_sentences)
print("Estimated average words per sentence:", avg_words_per_sentence)

# %%
# Reviews per user
reviews_per_user = df.groupby('user_id').size().reset_index(name='review_count')
print(reviews_per_user.describe())

print("\n")

# Reviews per item
reviews_per_item = df.groupby('parent_asin').size().reset_index(name='review_count')
print(reviews_per_item.describe())

print("\n")

reviews_90 = reviews_per_user['review_count'].quantile(0.90)
print("90% of the users have at most", reviews_90, "reviews.")

reviews_90 = reviews_per_item['review_count'].quantile(0.90)
print("90% of the items have at most", reviews_90, "reviews.")

# %%
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.hist(reviews_per_user['review_count'], bins=50, color='lightgreen', edgecolor='black')
plt.title("Distribution of Reviews per User")
plt.xlabel("Number of Reviews")
plt.ylabel("Frequency")
plt.xlim(0, 3)

plt.subplot(1, 2, 2)
plt.hist(reviews_per_item['review_count'], bins=50, color='lightblue', edgecolor='black')
plt.title("Distribution of Reviews per Item")
plt.xlabel("Number of Reviews")
plt.ylabel("Frequency")
plt.xlim(0, 200)

plt.tight_layout()
plt.show()

# %% [markdown]
# After a bit of experimenting, found out that we could take 3 reviews per user and 200 reviews per item.
# 
# This is with the graph, it was misleading. 90% of items have 34 reviews.

# %%
item_reviews = df.groupby('parent_asin')['text'].apply(lambda x: 'rrr'.join(x)).reset_index()
user_reviews= df.groupby('user_id')['text'].apply(lambda x: 'rrr'.join(x)).reset_index()

# %% [markdown]
# # Preprocessing into embeddings

# %%
fake_review_text = "This product is absolute shit! The surface looks like it's made by first grader, and comes apart pretty quickly. I highly recommend not buying it."

fake_embedding_dict = {
    "the": np.array([0.12, 0.34, 0.56, 0.78, 0.90]),
    "product": np.array([0.45, 0.67, 0.23, 0.89, 0.12]),
    "is": np.array([0.78, 0.12, 0.56, 0.34, 0.67]),
    "absolutely": np.array([0.89, 0.45, 0.32, 0.76, 0.21]),
    "amazing": np.array([0.34, 0.78, 0.91, 0.56, 0.23]),
    "characters": np.array([0.67, 0.23, 0.78, 0.45, 0.12]),
    "were": np.array([0.23, 0.56, 0.89, 0.12, 0.45]),
    "well-developed": np.array([0.56, 0.78, 0.34, 0.67, 0.90]),
    "and": np.array([0.91, 0.23, 0.45, 0.67, 0.78]),
    "plot": np.array([0.45, 0.12, 0.67, 0.89, 0.23]),
    "engaging": np.array([0.12, 0.67, 0.89, 0.34, 0.78]),
    "i": np.array([0.78, 0.34, 0.12, 0.56, 0.89]),
    "highly": np.array([0.23, 0.78, 0.45, 0.12, 0.67]),
    "recommend": np.array([0.67, 0.12, 0.56, 0.89, 0.45]),
    "watching": np.array([0.89, 0.23, 0.67, 0.78, 0.12]),
    "it": np.array([0.34, 0.56, 0.78, 0.12, 0.89]),
}


# %%
import gensim.downloader as downloader
embed_model = downloader.load('word2vec-google-news-300')
# To save the model for later use: model.save('google_news_embeddings.model')

# %%
vector = embed_model['product']
vector

# %%
def process_text(text, words_per_sentence, sentences_per_review, reviews_per_entity):

    raw_reviews = text.split("rrr")

    reviews = []
    for review in raw_reviews[:reviews_per_entity]:
        sentences = nltk.sent_tokenize(review)
        tokenized_sentences = []

        for sentence in sentences[:sentences_per_review]:
            words = nltk.word_tokenize(sentence)
            #words = sentence.split()

            words = words[:words_per_sentence] + ['<PAD>'] * max(0, words_per_sentence - len(words))
            tokenized_sentences.append(words)
        
        while len(tokenized_sentences) < sentences_per_review:  # sentences < sentences_per_review, pad with empty sentences (list of PAD tokens)
            tokenized_sentences.append(['<PAD>'] * words_per_sentence)
        reviews.append(tokenized_sentences)

    while len(reviews) < reviews_per_entity: # reviews < reviews_per_entity, pad with empty reviews.
        reviews.append([['<PAD>'] * words_per_sentence for _ in range(sentences_per_review)])
    return reviews

def text_to_embedding(tokenized_text, embedding_dict, embedding_dim):
    reviews_emb = []
    for review in tokenized_text:
        sentences_emb = []
        for sentence in review:
            sentence_emb = []
            for word in sentence:
                if word in embedding_dict:
                    emb = embedding_dict[word]
                else:
                    emb = np.zeros(embedding_dim)
                #emb = embedding_dict.get(word, np.zeros(embedding_dim))
                sentence_emb.append(emb)
            sentences_emb.append(sentence_emb)
        reviews_emb.append(sentences_emb)
    return torch.tensor(reviews_emb, dtype=torch.float)  # shape: (num_reviews, num_sentences, num_words, embedding_dim)

# %% [markdown]
# ### Alternate data loading method

# %%
# took 19.6 seconds for 105K rows

words_per_sentence = 12
sentences_per_review = 4
embedding_dim = 300
user_reviews_per_entity = 3
item_reviews_per_entity = 35

def process_text(text, sentences_per_review, words_per_sentence, embedding_dim, embed_dict):

    output = np.zeros((sentences_per_review, words_per_sentence, embedding_dim))
    sentences = nltk.sent_tokenize(str(text))
    
    for i, sentence in enumerate(sentences[0:sentences_per_review]):
        words = nltk.word_tokenize(sentence)
        for j, word in enumerate(words[0:words_per_sentence]):
            if word in embed_dict:
                output[i, j] = embed_dict[word]

    return torch.tensor(output)

df['embedding'] = df['text'].apply(lambda x: process_text(x, sentences_per_review, words_per_sentence, embedding_dim, embed_model))    

# %%
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import time

# %%

class RecSysDataset(Dataset):
    def __init__(self, df):
        self.df = df
        global words_per_sentence
        global sentences_per_review
        global embedding_dim
        global user_reviews_per_entity
        global item_reviews_per_entity
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        padding_tensor = torch.zeros((sentences_per_review, words_per_sentence, embedding_dim), dtype=torch.int)

        user_reviews = list(self.df[self.df['user_id'] == row['user_id']]['embedding'])[0:user_reviews_per_entity]
        user_emb = torch.stack(user_reviews + [padding_tensor] * (user_reviews_per_entity - len(user_reviews)))

        item_reviews = list(self.df[self.df['parent_asin'] == row['parent_asin']]['embedding'])[0:item_reviews_per_entity]
        item_emb = torch.stack(user_reviews + [padding_tensor] * (item_reviews_per_entity - len(user_reviews)))
        
        # user_emb = row['user_embedding']  # Tensor shape: (reviews_per_entity, sentences_per_review, words_per_sentence, embedding_dim)
        # item_emb = row['item_embedding']  # Same shape
        
        rating = row['rating']
        return user_emb, item_emb, torch.tensor(rating, dtype=torch.float)


batch_size = 32
dataset = RecSysDataset(df)

# num_workers argument is not working in Jupyter, but setting num_workers will improve the performance
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

start = time.time()

for sample, batch in tqdm(enumerate(dataloader)):
    user_emb, item_emb, rating = batch
    if sample > 100:
        break

end = time.time()
print(f'Time taken for 100 batches - {100*batch_size} records - {end-start}')

# %%
words_per_sentence = 12
sentences_per_review = 4
user_reviews_per_entity = 3
item_reviews_per_entity = 35
embedding_dim = 300

# %%
tokenized_item = process_text(fake_review_text, words_per_sentence, sentences_per_review, item_reviews_per_entity)
item_tensor = text_to_embedding(tokenized_item, fake_embedding_dict, embedding_dim)
# Now, item_tensor.shape should be (5, 5, 10, 50)

# %%
tokenized_item

# %%
print(item_tensor)

# %%
user_reviews['tokenized'] = user_reviews['text'].apply(
    lambda t: process_text(t, words_per_sentence, sentences_per_review, user_reviews_per_entity)
)
user_reviews['user_embedding'] = user_reviews['tokenized'].apply(
    lambda tokens: text_to_embedding(tokens, embed_model, embedding_dim)
)


item_reviews['tokenized'] = item_reviews['text'].apply(
    lambda t: process_text(t, words_per_sentence, sentences_per_review, item_reviews_per_entity)
)
item_reviews['item_embedding'] = item_reviews['tokenized'].apply(
    lambda tokens: text_to_embedding(tokens, embed_model, embedding_dim)
)

#till adding tokenized for items, it was 16s, while doing embedding, it last upto 1m 19s

# %%
user_emb_df = user_reviews[['user_id', 'user_embedding']]
item_emb_df = item_reviews[['parent_asin', 'item_embedding']]

df_reduced = df[['user_id', 'parent_asin', 'text', 'rating']]

merged = df_reduced.merge(user_emb_df, on='user_id', how='left')
merged = merged.merge(item_emb_df, on='parent_asin', how='left')

# %% [markdown]
# # Model

# %%
x = torch.randn(40, 5, 3)
x= x.transpose(1, 2)
#print(x)
convq= nn.Conv1d(in_channels= 3, out_channels= 4, kernel_size= 2)

print(convq)
#print(convq.weight)
print(convq.weight.shape)

output= convq(x)
print(output)



# %%
batch_size = 1
seq_len = 5
input_dim = 3
dk = 4

x = torch.randn(batch_size, seq_len, input_dim)
print("input shape: ", x.shape)
print("input:")
print(x)
print("\n")
print("transposed:")
print(x.transpose(1, 2))

# %%
convq= nn.Conv1d(in_channels= input_dim, out_channels= 4, kernel_size= 2)
print(convq)
print(convq.weight)
print(convq.weight.shape)

# %%
class SequenceEncodingModule(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, num_heads, max_seq_len):
        """
        Args:
            input_dim: Dimension of input embeddings, d.
            hidden_dim: Hidden dimension of this module, d_k.
            kernel_size: n (width for n-gram features)
            num_heads: Number of attention heads, H. (d_h = hidden_dim / num_heads)
            max_seq_len: Maximum sequence length (used for relative positional embeddings)
        """
        super(SequenceEncodingModule, self).__init__()
        self.input_dim = input_dim       # d - embedding dimension
        self.hidden_dim = hidden_dim     # d_k -  hidden layer dimension
        self.kernel_size = kernel_size   # n - ngram size
        self.num_heads = num_heads
        self.d_h = hidden_dim // num_heads  # subspace dimension d_h (assume hidden_dim divisible by num_heads)
        self.max_seq_len = max_seq_len

        # Convolution layers for computing q, k, v.
        # In PyTorch, Conv1d with kernel_size = n and padding = floor((n-1)/2) produces a sliding-window (concatenated) result.
        padding = kernel_size // 2  # floor((n-1)/2) works when n is odd.
        self.conv_q = nn.Conv1d(in_channels=input_dim, out_channels=hidden_dim, kernel_size=kernel_size, padding=padding)
        self.conv_k = nn.Conv1d(in_channels=input_dim, out_channels=hidden_dim, kernel_size=kernel_size, padding=padding)
        self.conv_v = nn.Conv1d(in_channels=input_dim, out_channels=hidden_dim, kernel_size=kernel_size, padding=padding)

        self.ff = nn.Linear(hidden_dim, hidden_dim)

        self.p_K = nn.Parameter(torch.randn(max_seq_len, max_seq_len, self.d_h))
        self.p_V = nn.Parameter(torch.randn(max_seq_len, max_seq_len, self.d_h))

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch, seq_len, input_dim)
        Returns:
            z: Tensor of shape (batch, seq_len, hidden_dim) after encoding.
        """
        batch, t, _ = x.size()
        # First, apply convolution. Conv1d expects (batch, channels, seq_len).
        x_conv = x.transpose(1, 2)  # (batch, input_dim, t)
        q = F.relu(self.conv_q(x_conv))  # (batch, hidden_dim, t)
        k = F.relu(self.conv_k(x_conv))
        v = F.relu(self.conv_v(x_conv))
        # Bring back to shape (batch, t, hidden_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        # Split each into H heads: reshape from (batch, t, hidden_dim) to (batch, H, t, d_h)
        q = q.view(batch, t, self.num_heads, self.d_h).transpose(1, 2)  # (batch, num_heads, t, d_h)
        k = k.view(batch, t, self.num_heads, self.d_h).transpose(1, 2)
        v = v.view(batch, t, self.num_heads, self.d_h).transpose(1, 2)

        # Standard attention scores: for each head h, for positions i and j: score_std[i,j] = <q_i^h, k_j^h>
        scores_std = torch.matmul(q, k.transpose(-2, -1))  # (batch, num_heads, t, t)

        # Relative position term for keys:
        # p_K has shape (max_seq_len, max_seq_len, d_h). Slice to current sequence length.
        p_K = self.p_K[:t, :t, :]  # (t, t, d_h) PyTorch is fucking weird.

        # extra scores: for each head h, each position i and j, e_relative[i,j] = <q_i^h, p_K[i,j]>
        # We do this by broadcasting: (batch, num_heads, t, 1, d_h) * (1, 1, t, t, d_h) -> sum over d_h.
        e_relative = (q.unsqueeze(3) * p_K.unsqueeze(0).unsqueeze(0)).sum(-1)  # (batch, num_heads, t, t)

        # Total attention scores:
        scores = (scores_std + e_relative) / math.sqrt(self.d_h)  # (batch, num_heads, t, t)
        attn = F.softmax(scores, dim=-1)  # attention weights a_{ij}^h

        # Computing attention output.
        # First, weighted sum for v:
        z_v = torch.matmul(attn, v)  # (batch, num_heads, t, d_h)
        # Now add relative contribution from values using p_V.
        p_V = self.p_V[:t, :t, :]  # (t, t, d_h)
        # For each i, add: sum_j a_{ij}^h * p_V^{ij}
        z_p = (attn.unsqueeze(-1) * p_V.unsqueeze(0).unsqueeze(0)).sum(dim=-2)  # (batch, num_heads, t, d_h)
        z = z_v + z_p  # (batch, num_heads, t, d_h)

        # Concatenate all heads: from (batch, num_heads, t, d_h) to (batch, t, hidden_dim)
        z = z.transpose(1, 2).contiguous().view(batch, t, self.hidden_dim)

        # feed-forward layer with ReLU (W_f and b_f in Equation (7))
        z = F.relu(self.ff(z))  # (batch, t, hidden_dim)
        return z

# %%
class SequenceAggregatingModule(nn.Module):
    def __init__(self, hidden_dim):

        #hidden_dim: d_k (the same hidden dimension used in encoding)

        super(SequenceAggregatingModule, self).__init__()
        self.hidden_dim = hidden_dim
        self.d_p = hidden_dim // 2  # per paper, d_p = d_k/2
        self.W_p = nn.Linear(hidden_dim, self.d_p)  # This covers multiplication by W_p and addition of b_p.
        # h is a learnable vector in R^(d_p)
        self.h = nn.Parameter(torch.randn(self.d_p))

    def forward(self, z):

        # intermediate representation: r_i = ReLU(W_p z_i + b_p)
        r = F.relu(self.W_p(z))  # (batch, seq_len, d_p)
        # importance scores: score_i = h^T r_i.
        scores = torch.matmul(r, self.h)  # (batch, seq_len)
        # Normalizing to obtain attention weights a_i.
        attn = F.softmax(scores, dim=-1)  # (batch, seq_len)
        # Weighted sum of z_i's.
        l = torch.sum(z * attn.unsqueeze(-1), dim=1)  # (batch, hidden_dim)
        return l

# %%
class SequenceEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, num_heads, max_seq_len):
        super(SequenceEncoder, self).__init__()
        self.encoding_module = SequenceEncodingModule(input_dim, hidden_dim, kernel_size, num_heads, max_seq_len)
        self.aggregating_module = SequenceAggregatingModule(hidden_dim)

    def forward(self, x):
        z = self.encoding_module(x)    # (batch, seq_len, hidden_dim)
        l = self.aggregating_module(z)  # (batch, hidden_dim)
        return l

# %% [markdown]
# # Training

# %%
class HierarchicalEncoder(nn.Module):
    def __init__(self, word_embedding_dim, hidden_dim, kernel_size, num_heads, words_per_sentence, sentences_per_review, reviews_per_entity):

        super(HierarchicalEncoder, self).__init__()

        self.sentence_encoder = SequenceEncoder(word_embedding_dim, hidden_dim, kernel_size, num_heads, words_per_sentence)
        self.review_encoder = SequenceEncoder(hidden_dim, hidden_dim, kernel_size, num_heads, sentences_per_review)
        self.final_encoder = SequenceEncoder(hidden_dim, hidden_dim, kernel_size, num_heads, reviews_per_entity)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (B, R, S, W, E) where:
                B: batch size,
                R: reviews per entity,
                S: sentences per review,
                W: words per sentence,
                E: word embedding dimension.
        Returns:
            final_rep: Tensor of shape (B, hidden_dim)
        """
        B, R, S, W, E = x.size()

        x_sentences = x.view(B * R * S, W, E) # Flatten reviews and sentences into one dimension: (B * R * S, W, E)
        sentence_reps = self.sentence_encoder(x_sentences) # Process all sentences at once: each sentence → a sentence representation (B * R * S, hidden_dim)
        sentence_reps = sentence_reps.view(B, R, S, -1) # Reshape back to (B, R, S, hidden_dim)

        x_reviews = sentence_reps.view(B * R, S, -1) # For each review, we have S sentence representations. Flatten reviews: (B * R, S, hidden_dim)
        review_reps = self.review_encoder(x_reviews) # Process all reviews: (B * R, hidden_dim)
        review_reps = review_reps.view(B, R, -1) # Reshape back to (B, R, hidden_dim)

        final_rep = self.final_encoder(review_reps) # Process the sequence of reviews for each entity: (B, R, hidden_dim) → (B, hidden_dim)
        return final_rep


class RatingPredictor(nn.Module):
    def __init__(self, rep_dim, hidden_size=64):
        super(RatingPredictor, self).__init__()
        self.fc1 = nn.Linear(rep_dim * 2, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)  # Predicts a single rating value

    def forward(self, user_rep, item_rep):
        x = torch.cat([user_rep, item_rep], dim=-1)
        x = torch.relu(self.fc1(x))
        rating = self.fc2(x)
        return rating

# complete recommendation model using a two-tower architecture.
class RecommendationModel(nn.Module):
    def __init__(self, word_embedding_dim, hidden_dim, kernel_size, num_heads, words_per_sentence,
                 sentences_per_review, user_reviews_per_entity, item_reviews_per_entity):
        super(RecommendationModel, self).__init__()
        self.user_encoder = HierarchicalEncoder(word_embedding_dim, hidden_dim, kernel_size, num_heads,
                                                  words_per_sentence, sentences_per_review, user_reviews_per_entity)
        self.item_encoder = HierarchicalEncoder(word_embedding_dim, hidden_dim, kernel_size, num_heads,
                                                  words_per_sentence, sentences_per_review, item_reviews_per_entity)
        self.rating_predictor = RatingPredictor(hidden_dim)

    def forward(self, user_input, item_input):
        # user_input and item_input are hierarchical tensors.
        user_rep = self.user_encoder(user_input)  # (batch, hidden_dim)
        item_rep = self.item_encoder(item_input)  # (batch, hidden_dim)
        rating = self.rating_predictor(user_rep, item_rep)  # (batch, 1)
        return rating

# %%
class RecSysDataset(Dataset):
    def __init__(self, df):
        self.df = df
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        user_emb = row['user_embedding']  # Tensor shape: (reviews_per_entity, sentences_per_review, words_per_sentence, embedding_dim)
        item_emb = row['item_embedding']  # Same shape
        rating = row['rating']
        # making sure rating is a float tensor.
        return user_emb, item_emb, torch.tensor(rating, dtype=torch.float)

batch_size = 32
dataset = RecSysDataset(merged)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# %%
word_embedding_dim = 300
hidden_dim = 128
kernel_size = 3
num_heads = 4
words_per_sentence = 12
sentences_per_review = 4
user_reviews_per_entity = 3
item_reviews_per_entity = 35


model = RecommendationModel(word_embedding_dim, hidden_dim, kernel_size, num_heads, words_per_sentence,
                            sentences_per_review, user_reviews_per_entity, item_reviews_per_entity)

optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

# %%
model.summary()

# %%

num_epochs = 5

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0
    for batch in dataloader:
        user_input, item_input, ratings = batch
        # user_input, item_input: shape (batch, reviews_per_entity, sentences_per_review, words_per_sentence, embedding_dim)
        optimizer.zero_grad()
        predictions = model(user_input, item_input).squeeze(-1)  # (batch,)
        loss = criterion(predictions, ratings)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    avg_loss = epoch_loss / len(dataloader)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")


