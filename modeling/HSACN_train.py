import os
import time
import yaml
import torch
import wandb

import random
import numpy as np

import torch.nn as nn
import torch.optim as optim

from torchinfo import summary
from HSACN import HSACN
import final_dataset_HSACN as dataset



def seed_everything(seed=42):
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    torch.backends.cudnn.deterministic = True
    # this may hurt the performance
    torch.backends.cudnn.benchmark = False

def train(model, run):
    model.train()
    batches_trained = 0 #global batch no. for all epochs
    
    for i in range(epochs):

        for index, (user_tower_input, item_tower_input, true_rating) in enumerate(train_dataloader):
            
            """if batches_trained == max_iters:
                break"""
            
            user_tower_input, item_tower_input, true_rating = user_tower_input.to(device), item_tower_input.to(device), true_rating.to(device)

            optimizer.zero_grad(set_to_none=True)
            predicted_rating = model(user_tower_input, item_tower_input)
            loss = loss_fn(predicted_rating, true_rating.view(-1, 1))
            metrics = {'train/train_loss': loss.item()}
            
            loss.backward()
            optimizer.step()
            #print(f'Epoch - {i} | step - {index} | train loss - {loss:.4f}')
            
            if batches_trained % eval_iters != 0 and index != len(train_dataloader) - 1:
                run.log(metrics)
            
            if batches_trained % eval_iters == 0 and index != len(train_dataloader) - 1 and batches_trained!=0 : #to calculate in between epoch after a few iters
                val_loss = evaluate(model, val_dataloader)
                metrics['val/val_loss'] = val_loss
                print(f'Epoch - {i} | step - {index} | train loss - {loss:.4f} | val loss - {val_loss:.4f}')
                run.log(metrics)
            
            batches_trained += 1
        
        #to calculate after every epoch
        val_loss = evaluate(model, val_dataloader)
        metrics['val/val_loss'] = val_loss
        print(f'Epoch - {i} | step - {index} | train loss - {loss:.4f} | val loss - {val_loss:.4f}')
        run.log(metrics)


@torch.no_grad()
def evaluate(model, dataloader):
    model.eval()
    total_error = 0.0
    batches_evaluated = 0
    
    for index, (user_tower_input, item_tower_input, true_rating) in enumerate(dataloader):
        
        # use this only for big data, not while using the sampled one
        """if batches_evaluated >= eval_batches:
            break"""

        user_tower_input, item_tower_input, true_rating = user_tower_input.to(device), item_tower_input.to(device), true_rating.to(device)
        predicted_rating = model(user_tower_input, item_tower_input)
        error = loss_fn(predicted_rating, true_rating.view(-1, 1))
        total_error += error.item()
        
        batches_evaluated += 1
    
    avg_error = total_error / eval_batches
    return avg_error


if __name__ == '__main__':

    with open('hsacn.yaml') as f:
        config = yaml.safe_load(f)

    wandb_entity = "kodati-sr-northeastern-university"
    wandb_project = "RecSys"
    wandb_run_name = config['l']["wandb_run_name"]
    wandb_tags = ["hsacn", wandb_run_name, "train"]
    wandb_model = wandb_run_name + "_model"
    wandb_model_version = "v0"

    retrain = False
    load_local_ckpt = False
    local_ckpt_path = "checkpoints/" + f"{wandb_model}_checkpoint_final.pt"
    wandb_existing_run_name = "retrain_test_1"

    api = wandb.Api()
    save_dir = 'checkpoints'
    os.makedirs(save_dir, exist_ok=True)


    if retrain:
        # fetch run id; using run id fetch the config details and download the model
        runs = api.runs(f"{wandb_entity}/{wandb_project}")
        run_id = next((run.id for run in runs if run.name == wandb_existing_run_name), None)
        existing_run = api.run(f"{wandb_entity}/{wandb_project}/{run_id}")
        key = int(list(existing_run.config.keys())[-1]) + 1
        # updated config is loaded with new key
        existing_run.config[str(key)] = config
        
        run = wandb.init(entity=wandb_entity, project=wandb_project, id=run_id, resume="must")
        if not load_local_ckpt:
            artifact = run.use_artifact(f'{wandb_entity}/{wandb_project}/{wandb_model}:{wandb_model_version}', type='model')
            artifact_dir = artifact.download(root=save_dir)
    else:
        initial_config = {'1':config}
        run = wandb.init(project=wandb_project, 
                    config=initial_config, 
                    tags=wandb_tags,
                    name=wandb_run_name + '_' + time.ctime())

    epochs = config['t']['num_epochs']
    eval_iters = config['t']['eval_iters']
    eval_batches = config['t']['eval_batches']
    checkpoint_iters = config['t']['checkpoint_iters']
    max_iters = config['t']['max_iters']
    batch_size = config['t']['batch_size']

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")  # macOS Metal Performance Shaders
    else:
        device = torch.device("cpu")

    seed_everything(42)
    
    word_embedding_dim = config['m']['word_embedding_dim']
    words_per_sentence = config['m']['words_per_sentence']
    sentences_per_review = config['m']['sentences_per_review']   
    user_reviews_per_entity = config['m']['user_reviews_per_entity']    
    item_reviews_per_entity = config['m']['item_reviews_per_entity'] 
    num_heads = config['m']['num_heads']   
    kernel_size = config['m']['kernel_size'] 
    hidden_dim = config['m']['hidden_dim'] 

    glove_embeddings = torch.load('required_embeddings.pt').to(torch.float32)

    model = HSACN(word_embedding_dim, hidden_dim, kernel_size, num_heads, words_per_sentence,
                            sentences_per_review, user_reviews_per_entity, item_reviews_per_entity, glove_embeddings)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=config['t']['learning_rate'])

    if retrain:
        ckpt = torch.load(local_ckpt_path)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    loss_fn = nn.MSELoss()

    summary(model)

    train_dataloader, val_dataloader, test_dataloader = dataset.train_dataloader, dataset.val_dataloader, dataset.test_dataloader
    train(model, run)

    test_loss = evaluate(model, test_dataloader)
    run.summary['test/test_loss'] = test_loss
    print(f'Final test loss - {test_loss}')

    checkpoint_path = os.path.join(save_dir, f"{wandb_model}_checkpoint_final.pt")

    torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, checkpoint_path)

    run.log_model(f"./checkpoints/{wandb_model}_checkpoint_final.pt", name=wandb_model)
    run.finish()