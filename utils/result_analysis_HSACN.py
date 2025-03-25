import yaml
import torch
import pandas as pd
import torch.nn as nn
import torch.optim as optim

from HSACN import HSACN
from final_dataset_HSACN import val_df, val_dataloader

predicted_ratings = []
true_ratings = []
user_ids = []
item_ids = []

with open('hsacn.yaml') as f:
    config = yaml.safe_load(f)

num_users= config['m']['num_users']
num_items= config['m']['num_items']
word_embedding_dim = config['m']['word_embedding_dim']
words_per_sentence = config['m']['words_per_sentence']
sentences_per_review = config['m']['sentences_per_review']   
user_reviews_per_entity = config['m']['user_reviews_per_entity']    
item_reviews_per_entity = config['m']['item_reviews_per_entity'] 
num_heads = config['m']['num_heads']   
kernel_size = config['m']['kernel_size'] 
hidden_dim = config['m']['hidden_dim']
latent_dim = config['m']['latent_dim'] 


glove_embeddings = torch.load('required_embeddings.pt').to(torch.float32)

model = HSACN(num_users, num_items, word_embedding_dim, hidden_dim, latent_dim, kernel_size, num_heads, words_per_sentence,
                        sentences_per_review, user_reviews_per_entity, item_reviews_per_entity, glove_embeddings)
optimizer = optim.Adam(model.parameters(), lr=config['t']['learning_rate'])

checkpoint = torch.load("/content/train_final_fr_model_checkpoint_final.pt")

model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
model.eval()

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")  # macOS Metal Performance Shaders
else:
    device = torch.device("cpu")

model = model.to(device)
loss_fn = nn.MSELoss()

total_error = 0
batches_evaluated = 0

with torch.no_grad():
    for user_tower_input, item_tower_input, true_rating, user_id, item_id in val_dataloader:
        user_tower_input, item_tower_input, true_rating, user_id, item_id = user_tower_input.to(device), item_tower_input.to(device), true_rating.to(device), user_id.to(device), item_id.to(device)
        predicted_rating = model(user_id, item_id, user_tower_input, item_tower_input)
        predicted_rating = torch.clamp(predicted_rating, min=1.0, max=5.0)


        error = loss_fn(predicted_rating, true_rating.view(-1, 1))
        total_error += error.item()
        batches_evaluated +=1

        predicted_ratings.extend(predicted_rating.cpu().numpy().flatten())
        true_ratings.extend(true_rating.cpu().numpy().flatten())
        user_ids.extend(user_id.cpu().numpy().flatten())
        item_ids.extend(item_id.cpu().numpy().flatten())
    
    avg_error = total_error / batches_evaluated

results_df = pd.DataFrame({
    'user_id': user_ids,
    'parent_asin': item_ids,
    'true_rating': true_ratings,
    'predicted_rating': predicted_ratings
})

merged_df = pd.merge(val_df.reset_index(drop=True), results_df, on=['user_id', 'parent_asin'])
merged_df.to_csv('merged_df.csv', index=False)

print("avg MSE: ", avg_error)