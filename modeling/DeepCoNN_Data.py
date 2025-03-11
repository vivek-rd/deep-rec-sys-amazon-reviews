import time
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gensim.downloader as downloader
from tensorflow.keras.preprocessing.text import text_to_word_sequence
from nltk import word_tokenize
from torch.utils.data import Dataset, DataLoader, TensorDataset
from collections import defaultdict


def load_data(path):
    return pd.read_csv(path)

def preprocess_data(df):
    df = df.loc[:,['user_id','parent_asin','text','rating']]
    df['text'] = df['text'].astype(str)
    df['text'] = df['text'].apply(lambda x: ' '.join(text_to_word_sequence(x)))
    df = df[df['text'].str.len() > 1]
    df.drop_duplicates(subset=['user_id','parent_asin'], inplace=True)     # Remove the duplicate user_id-asin pairs      
    # Only keep the unique user_id-asin pairs
    df.dropna(inplace=True)
    df['row_idx'] = df.index 
    return df


# Get All Reviews for a User except the review in the current row
def get_user_reviews(df, u_id, row_id):
    temp_df = df.groupby('user_id').get_group(u_id)
    temp_df = temp_df[temp_df['row_idx']!=row_id]
    user_review = list(temp_df['text'])
    return ' <SEP> '.join(user_review)


# Get All Reviews for an Item except the review in the current row
def get_item_reviews(df, i_id, row_id):
    temp_df = df.groupby('parent_asin').get_group(i_id)
    temp_df = temp_df[temp_df['row_idx']!=row_id]
    item_review = list(temp_df['text'])
    return ' <SEP> '.join(item_review)


# Pre-Trained Word-2-Vec Google News with 3,000,000 words and embedding dimensions 300
embed_model = downloader.load('word2vec-google-news-300')
embed_dim = embed_model.vector_size
embed_model['<SEP>'] = np.random.randn(embed_dim)
embed_model['<UNK>'] = np.random.randn(embed_dim)

# Create Embeddings for User and Item Reviews
def create_embeddings(text, max_length):
    tokens = word_tokenize(text)
    if len(tokens)>max_length:
        tokens=tokens[:max_length]
    
    embeddings = np.zeros((max_length, embed_dim))
    for idx, word in enumerate(tokens):
        if word in embed_model:
            embeddings[idx] = embed_model[word]
        else:
            embeddings[idx] = embed_model['<UNK>']
    return torch.from_numpy(embeddings).to(torch.float32)


# PyTorch Dataset Class to create data format required for Neural Network Training
class DeepConnDataset(Dataset):
    def __init__(self, df, max_user_review_length, max_item_review_length):
        self.df = df
        self.max_user_review_length = max_user_review_length
        self.max_item_review_length = max_item_review_length

        # Pre-grouping user and item reviews instead of fetching it everytime in getitem
        self.user_reviews = defaultdict(list)
        self.item_reviews = defaultdict(list)

        for _, row in df.iterrows():
            self.user_reviews[row['user_id']].append((row['parent_asin'], row['text']))
            self.item_reviews[row['parent_asin']].append((row['user_id'], row['text']))
        

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        u_id = torch.tensor(row['user_id'], dtype=torch.long)
        i_id = torch.tensor(row['parent_asin'], dtype=torch.long)
        rating = torch.tensor(row['rating'], dtype=torch.float32)

        # Pre-grouped texts, first filtering and then concatenating, makes a lot of difference in computation
        #rev is a tuple which has id, review text. id is to make sure to exclude the right review, as we may risk taking out same user giving same review to another item
        user_review = ' <SEP> '.join([rev[1] for rev in self.user_reviews[u_id] if rev[0] != i_id])
        item_review = ' <SEP> '.join([rev[1] for rev in self.item_reviews[i_id] if rev[0] != u_id])

        # user_review = get_user_reviews(self.df, row['user_id'], row['row_idx'])
        # item_review = get_item_reviews(self.df, row['parent_asin'], row['row_idx'])
        
        user_embeddings = create_embeddings(user_review, self.max_user_review_length)
        item_embeddings = create_embeddings(item_review, self.max_item_review_length)
        return u_id, i_id, user_embeddings, item_embeddings, rating
    

# Fetch train, val and test datasets
# train_df = load_data("/Users/rutvikdhopate/Downloads/train_df_with_text.csv")
# val_df = load_data("/Users/rutvikdhopate/Downloads/val_df_with_text.csv")
# test_df = load_data("/Users/rutvikdhopate/Downloads/test_df_with_text.csv")

train_df = load_data("data/new_train_df.csv")
val_df = load_data("data/new_val_df.csv")
test_df = load_data("data/new_test_df.csv")

train_df = preprocess_data(train_df)
val_df = preprocess_data(val_df)
test_df = preprocess_data(test_df)

user_ids = set(train_df['user_id'].dropna().unique())
item_ids = set(train_df['parent_asin'].dropna().unique())

user_codes = pd.CategoricalDtype(user_ids)
item_codes = pd.CategoricalDtype(item_ids)

for df in [train_df, val_df, test_df]:
    df['user_id'] = df['user_id'].astype(user_codes).cat.codes
    df['parent_asin'] = df['parent_asin'].astype(item_codes).cat.codes


train_dataset = DeepConnDataset(train_df, max_user_review_length=200, max_item_review_length=1000)
val_dataset = DeepConnDataset(val_df, max_user_review_length=200, max_item_review_length=1000)
test_dataset = DeepConnDataset(test_df, max_user_review_length=200, max_item_review_length=1000)

# Hyperparameters
max_review_length_u = 200
max_review_length_i = 1000
embed_dim = 300
t = [3, 5]                  # Kernel Width
n1 = 100                    # Kernel Depth
latent_factors = 50 
fm_k = 10           # Number of factors in Factorization Machine
reg_lambda = 2e-3   # Regularization Lambda
batch_size = 32
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

# Batches
start = time.time()
for i, batch in enumerate(train_dataloader):
    u_id, i_id, u_embed, i_embed, rating = batch
    if i == 10:
        break

end = time.time()
print(f"Time taken for embedding - {(end-start)/60} mins")


# DeepCoNN Architecture
# Convolution Max-Pooling Layer
class ConvMaxLayer(torch.nn.Module):
    '''
        The independent layer for user review and item review
    '''
    def __init__(self, max_review_length, t, embed_dim, n1, latent_factors):
        super().__init__()
        self.max_review_length = max_review_length
        self.t = t
        self.embed_dim = embed_dim
        self.n1 = n1
        self.latent_factors = latent_factors

        self.convs = torch.nn.ModuleList()
        self.maxs = torch.nn.ModuleList()

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

        self.activation = torch.nn.ReLU()       # Shared activation function for user and item
        self.full_connect = torch.nn.Linear(self.n1 * len(self.t), self.latent_factors)     # Shared fully connected layer

    
    def forward(self, review):
        """
            Input Shape: (Batch Size, Review Length, Word Embedding Size)
            Output Shape: (Batch Size, Latent Factors Size)
        """
        output = []
        review = review.permute(0,2,1)
        for max_pool, conv in zip(self.maxs, self.convs):
            out = self.activation(conv(review))
            max_out = max_pool(out)
            flatten_out = torch.flatten(max_out, start_dim=1)        # Passed stride=1 as an argument and it returned an error
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
        out_lin = self.lin(x)       # Got matmul error (32x1 and 100x1) so changed out_inter to x
        out = out_inter + out_lin
        return out
    

class DeepCoNN(nn.Module):
    def __init__(self, max_review_length_u, max_review_length_i, t, embed_dim, n1, latent_factors, fm_k):
        super().__init__()

        self.max_review_length_u = max_review_length_u
        self.max_review_length_i = max_review_length_i
        self.t = t
        self.embed_dim = embed_dim
        self.n1 = n1
        self.latent_factors = latent_factors
        self.fm_k = fm_k


        # self.embedding = torch.nn.Embedding.from_pretrained(embedding_weight)
        # self.embedding.weight.requires_grad = False

        self.user_layer = ConvMaxLayer(max_review_length_u, t, embed_dim, n1, latent_factors)
        self.item_layer = ConvMaxLayer(max_review_length_i, t, embed_dim, n1, latent_factors)
        self.share_layer = FMLayer(latent_factors, fm_k)

    def forward(self, user_review, item_review):
        """
            Input Shape: (Batch Size, Review Length)
            Output ShapeL (Batch Size)
        """

        user_latent = self.user_layer(user_review)
        item_latent = self.item_layer(item_review)
        latent = torch.cat([user_latent, item_latent], dim=1)
        predict = self.share_layer(latent)
        return predict

print("Checkpoint - Architecture Done")


# Training Loop
model = DeepCoNN(max_review_length_u=200, max_review_length_i=1000, t=t, embed_dim=embed_dim, n1=n1, latent_factors=latent_factors, fm_k=fm_k)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=reg_lambda)

# Evaluate Loss
@torch.no_grad()
def evaluate(model, dataloader):
    model.eval()
    total_loss = 0
    batches_evaluated = 0
    for i, batch in enumerate(dataloader):
        u_id, i_id, u_embed, i_embed, rating = batch
        u_embed, i_embed, rating = u_embed.to(device), i_embed.to(device), rating.to(device)
        predictions = model(u_embed, i_embed).squeeze()
        loss = criterion(predictions, rating)
        print(f"Batch Number - {i+1}, Batch Loss = {loss.item():.4f}")
        total_loss+=loss.item()
        batches_evaluated+=1
    avg_total_loss = total_loss/batches_evaluated
    return avg_total_loss



start = time.time()
print(start)
num_epochs = 1
epoch_losses = {i:None for i in range(num_epochs)}
best_val_loss = float('inf')
max_batches = 1000

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    batch_losses = []
    for i, batch in enumerate(train_dataloader):
        if i >= max_batches:        # Remove While Full Training
            break
        u_id, i_id, u_embed, i_embed, rating = batch
        u_embed, i_embed, rating = u_embed.to(device), i_embed.to(device), rating.to(device)

        optimizer.zero_grad()
        predictions = model(u_embed, i_embed).squeeze()
        loss = criterion(predictions, rating)
        loss.backward()
        optimizer.step()
        total_loss+=loss.item()
        
        batch_losses.append(loss.item())
        if i%50==0:
            print(f"Epoch: {epoch+1}/{num_epochs}, Batch: {i+1}, Batch Train Loss: {loss.item():.4f}")

    epoch_losses[i] = batch_losses
    avg_train_loss = total_loss/len(train_dataloader)
    
    # Validation Loop
    avg_val_loss = evaluate(model, val_dataloader)
    
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        # Save the model here
        torch.save(model.state_dict(), "deepconn_train_iter1.pth")
        print("Best Validation Loss - Model Saved")


    print(f"Epoch: {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
end = time.time()
print(f"Time taken for embedding - {(end-start)/60} mins")


print("Test Evaluation")
# Test Evaluation
avg_test_loss = evaluate(model, test_dataloader)
print(f"Average Test Loss: {avg_test_loss:.4f}")