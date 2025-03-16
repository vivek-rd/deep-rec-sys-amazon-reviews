import time
import nltk
import torch
import random
import argparse
import numpy as np
import pandas as pd

from torch.utils.data import DataLoader, Dataset
import gensim.downloader as downloader

def process_list(text_list, max_reviews, words_per_sentence, model='NRCMA', sentences_per_review=4):
    """
    Takes in a list of reviews - i.e pre-grouped strings and outputs final vector
    """
    
    if model == 'NRCMA':
        final_output = np.zeros((max_reviews, words_per_sentence))
        
        for i, text in enumerate(text_list[:max_reviews]):
            words = nltk.word_tokenize(str(text))
            # Following same logic when creating vocabulary, I think additional logic can be
            # added here to get more tokens into vectors - split tokens like "five-star" into five, star which would be present in the list 
            for j, word in enumerate(words[:words_per_sentence]):
                if word in word_to_index.keys():
                    final_output[i, j] = word_to_index[word]
                elif word.lower() in word_to_index.keys():
                    final_output[i, j] = word_to_index[word.lower()]
                else:
                    final_output[i, j] = word_to_index['<unk>']
    
    if model == "HSACN":
        final_output = np.zeros((max_reviews, sentences_per_review, words_per_sentence))
        
        for i, text in enumerate(text_list[0:max_reviews]):
            sentences = nltk.sent_tokenize(str(text))
            for j, sentence in enumerate(sentences[0:sentences_per_review]):
                words = nltk.word_tokenize(str(sentence))
                for k, word in enumerate(words[0:words_per_sentence]):
                    if word in word_to_index.keys():
                        final_output[i, j, k] = word_to_index[word]
                    elif word.lower() in word_to_index.keys():
                        final_output[i, j, k] = word_to_index[word.lower()]
                    else:
                        final_output[i, j, k] = word_to_index['<unk>']
                

    return torch.tensor(final_output, dtype=torch.int)  # Fixed: returning correct tensor


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", choices=['NRCMA', 'HSACN'], required=True)

    args = parser.parse_args()
    model = args.model # model can be NRCMA or HSACN
    print(f'Creating embeddings for - {model}')
    
    train_df = pd.read_csv('train_df_filtered.csv')
    val_df = pd.read_csv('val_df_filtered.csv')
    test_df = pd.read_csv('test_df_filtered.csv')
    
    total_text = ' '.join(list(train_df['text'].astype('str')))
    unique_words = set(nltk.word_tokenize(total_text))
    print(f'Number of unique words in train df - {len(unique_words)}')
    
    embed_model = downloader.load("word2vec-google-news-300")
    
    embedding_dim = 300
    unk_vector = np.random.randn(embedding_dim)
    embed_model.add_vector('<unk>', unk_vector)  # Add unknown token
    
    required_glove_words = set() # Words for which we need to pick the glove vectors
    missing_words = set()

    # Follow the same logic while converting the nltk tokens into indices
    for word in unique_words:
        if word in embed_model:
            required_glove_words.add(word)
        elif word.lower() in embed_model:
            required_glove_words.add(word.lower())
        else:
            missing_words.add(word)
    
    # +2 for pad and unk tokens
    print(f'Final vocabulary size - {len(required_glove_words)+2}, number of missing words - {len(missing_words)}')
    print(f'Random sample of missing words - {random.sample(sorted(missing_words), 100)}')
    
    word_to_index = {}
    word_to_index['<pad>'] = 0
    for index, word in enumerate(required_glove_words):
        word_to_index[word] = index + 1 # adding 1 because <pad> is 0th index so we need to start index variable at 1
    word_to_index['<unk>'] = len(word_to_index) # unk is the final index
    
    index_to_word = {value:key for key, value in word_to_index.items()}
    vocab_size = len(word_to_index)
    
    required_embeddings = np.zeros((vocab_size, embedding_dim))
    
    for index, word in enumerate(list(word_to_index.keys())[1:]): # skipping the first element as pad token is already set to zeros
        required_embeddings[index, :] = embed_model[word]
    
    # save the required_embeddings, word_to_index, index_word
    required_embeddings = torch.from_numpy(required_embeddings)
    torch.save(required_embeddings, 'required_embeddings.pt')
    
    user_ids = set(train_df['user_id'].dropna().unique())
    item_ids = set(train_df['parent_asin'].dropna().unique())

    user_codes = pd.CategoricalDtype(user_ids)
    item_codes = pd.CategoricalDtype(item_ids)

    for df in [train_df, val_df, test_df]:
        df['user_id'] = df['user_id'].astype(user_codes).cat.codes
        df['parent_asin'] = df['parent_asin'].astype(item_codes).cat.codes

    # Group users' reviews
    user_groups = train_df.groupby('user_id').indices  # dict: user_id -> list of row indices
    item_groups = train_df.groupby('parent_asin').indices # dict: parent_asin -> list of row indices
    
    max_user_reviews, max_item_reviews, words_per_sentence = 3, 35, 12
    sentences_per_review = 4 # for HSACN
    
    val_df['train_user_idx'] = val_df.apply(lambda row: np.where(train_df['user_id'] == row['user_id'])[0][0], axis=1)
    val_df['train_item_idx'] = val_df.apply(lambda row: np.where(train_df['parent_asin'] == row['parent_asin'])[0][0], axis=1)
    
    test_df['train_user_idx'] = test_df.apply(lambda row: np.where(train_df['user_id'] == row['user_id'])[0][0], axis=1)
    test_df['train_item_idx'] = test_df.apply(lambda row: np.where(train_df['parent_asin'] == row['parent_asin'])[0][0], axis=1)
    
    if model == 'NRCMA':
        
        user_tower_inputs = torch.zeros((len(train_df)), max_user_reviews, words_per_sentence)
        item_tower_inputs = torch.zeros((len(train_df)), max_item_reviews, words_per_sentence)

        for i, row in train_df.iterrows():
            all_user_indices = user_groups[row['user_id']]
            all_item_indices = user_groups[row['parent_asin']]
            
            available_user_indices = [idx for idx in all_user_indices if idx != i] # excluding current review
            available_item_indices = [idx for idx in all_item_indices if idx != i] # excluding current review
            
            user_text = list(train_df.loc[available_user_indices, 'text'])
            item_text = list(train_df.loc[available_item_indices, 'text'])
            
            user_tower_inputs[i] = process_list(user_text, max_user_reviews, words_per_sentence, model=model)
            item_tower_inputs[i] = process_list(item_text, max_item_reviews, words_per_sentence, model=model)
    
    elif model == 'HSACN':
        
        user_tower_inputs = torch.zeros((len(train_df)), max_user_reviews, sentences_per_review, words_per_sentence)
        item_tower_inputs = torch.zeros((len(train_df)), max_item_reviews, sentences_per_review, words_per_sentence)

        for i, row in train_df.iterrows():
            all_user_indices = user_groups[row['user_id']]
            all_item_indices = user_groups[row['parent_asin']]
            
            available_user_indices = [idx for idx in all_user_indices if idx != i] # excluding current review
            available_item_indices = [idx for idx in all_item_indices if idx != i] # excluding current review
            
            user_text = list(train_df.loc[available_user_indices, 'text'])
            item_text = list(train_df.loc[available_item_indices, 'text'])
            
            user_tower_inputs[i] = process_list(user_text, max_user_reviews, words_per_sentence, model=model, sentences_per_review=sentences_per_review)
            item_tower_inputs[i] = process_list(item_text, max_item_reviews, words_per_sentence, model=model, sentences_per_review=sentences_per_review)
    
    torch.save(user_tower_inputs, f'user_tower_input_{model}.pt')
    torch.save(item_tower_inputs, f'item_tower_input_{model}.pt')
    
    train_df.to_csv('train_df_filtered.csv', index=False)
    val_df.to_csv('val_df_filtered.csv', index=False)
    test_df.to_csv('test_df_filtered.csv', index=False)
    
    print(f'saved files succesfully!')
    
        
    
    
    