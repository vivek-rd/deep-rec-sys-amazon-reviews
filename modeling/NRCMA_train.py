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
from modeling.NRCMA import NRCMA, NRCMAConfig
from utils import final_dataset


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
    batches_trained = 0
    
    for i in range(epochs):

        for index, batch in enumerate(train_dataloader):
            
            if max_iters != 'None' and batches_trained == max_iters:
                break
            
            optimizer.zero_grad(set_to_none=True)
            user_tower_input, item_tower_input, true_rating, user, item, _ = batch
            user_tower_input, item_tower_input, true_rating, user, item = user_tower_input.to(device), \
            item_tower_input.to(device), true_rating.to(device), user.to(device), item.to(device) 
            predicted_rating = model(user_tower_input, item_tower_input, user, item)
            loss = loss_fn(predicted_rating, true_rating)
            # Cap predictions between 1 and 5
            # predicted_rating = torch.clamp(predicted_rating, min=1.0, max=5.0)
            # loss_per_sample = loss_fn(predicted_rating, true_rating)
            # weights = torch.where(true_rating <= 3, 
            #           torch.tensor(2.0, device=true_rating.device), 
            #           torch.tensor(1.0, device=true_rating.device))
            
            # # Multiply each sample's loss by its weight and take the mean
            # loss = (loss_per_sample * weights).mean()
            
            # if index != 0:
            metrics = {'train/train_loss': loss.item()}
            # else:
            #     metrics['train/train_loss'] = loss.item()
            
            loss.backward()
            optimizer.step()
            
            if batches_trained % eval_iters != 0 and index != len(train_dataloader) - 1:
                run.log(metrics)
                print(f'Epoch - {i} | step - {index} | train loss - {loss:.4f}')
            
            if batches_trained % eval_iters == 0 and index != len(train_dataloader) - 1 and batches_trained != 0 :
                val_loss = evaluate(model, val_dataloader)
                metrics['val/val_loss'] = val_loss
                print(f'Epoch - {i} | step - {index} | train loss - {loss:.4f} | val loss - {val_loss:.4f}')
                run.log(metrics)
            
            batches_trained += 1
        
        val_loss = evaluate(model, val_dataloader, print_predictions=True)
        metrics['val/val_loss'] = val_loss
        print(f'Epoch - {i} | step - {index} | train loss - {loss:.4f} | val loss - {val_loss:.4f}')
        run.log(metrics)


@torch.no_grad()
def evaluate(model, dataloader, print_predictions=False):
    model.eval()
    total_error = 0.0
    batches_evaluated = 0
    
    for index, batch in enumerate(dataloader):
        
        if eval_batches != 'None' and batches_evaluated >= eval_batches:
            break

        user_tower_input, item_tower_input, true_rating, user, item = batch
        user_tower_input, item_tower_input, true_rating, user, item = user_tower_input.to(device), \
            item_tower_input.to(device), true_rating.to(device), user.to(device), item.to(device) 
        predicted_rating = model(user_tower_input, item_tower_input, user, item)
        # Cap predictions between 1 and 5
        predicted_rating = torch.clamp(predicted_rating, min=1.0, max=5.0)
        error = loss_fn(predicted_rating, true_rating)
        
        # error = loss_fn(predicted_rating, true_rating).mean()
        # loss_per_sample = loss_fn(predicted_rating, true_rating)
        # weights = torch.where(true_rating <= 3, 
        #             torch.tensor(2.0, device=true_rating.device), 
        #             torch.tensor(1.0, device=true_rating.device))
            
        # # Multiply each sample's loss by its weight and take the mean
        # loss = (loss_per_sample * weights).mean()
        
        total_error += error.item()
        
        if print_predictions:
            # Determine the sample count: first 5 records (or fewer if batch is smaller)
            sample_count = min(5, true_rating.size(0))
            # Use slicing to vectorize the selection of predictions and actual ratings
            sampled_preds = predicted_rating[:sample_count].cpu().numpy()
            sampled_true = true_rating[:sample_count].cpu().numpy()
            print(f"Batch {batches_evaluated} predictions (first {sample_count} records):")
            print("Predicted ratings:", sampled_preds)
            print("Actual ratings:   ", sampled_true) 
        
        batches_evaluated += 1
    
    avg_error = total_error / batches_evaluated
    model.train()
    return avg_error


if __name__ == '__main__':

    wandb_entity = "rvivek-northeastern-university"
    wandb_project = "deep_rec_sys_user_reviews"
    wandb_run_name = "self_attention_word_embedding_tuning"
    wandb_tags = ["self_attention", "best_test_loss"]
    wandb_model = wandb_run_name + "_model"
    wandb_model_version = "v0"

    retrain = False
    load_local_ckpt = True
    local_ckpt_path = "checkpoints/" + f"{wandb_model}_checkpoint_final.pt"
    wandb_existing_run_name = "retrain_test_3_Sun Mar  9 12:55:51 2025"

    api = wandb.Api()
    save_dir = 'checkpoints'

    # load the current config file
    with open('config/nrcma.yaml') as f:
        config = yaml.safe_load(f)

    if retrain:
        # fetch run id; using run id fetch the config details and download the model
        runs = api.runs(f"{wandb_entity}/{wandb_project}")
        run_id = next((run.id for run in runs if run.name == wandb_existing_run_name), None)
        existing_run = api.run(f"{wandb_entity}/{wandb_project}/{run_id}")
        key = int(list(existing_run.config.keys())[-1]) + 1
        # updated config is loaded with new key
        existing_run.config[str(key)] = config
        
        run = wandb.init(entity=wandb_entity, project=wandb_project, id=run_id, resume="must", config=existing_run.config)
        print(f'Resuming training for existing checkpoint')
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

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")  # macOS Metal Performance Shaders
    else:
        device = torch.device("cpu")

    seed_everything(42)
    nrcma_config = NRCMAConfig.from_config(config['m'])
    glove_embeddings = torch.load('data/required_embeddings.pt').to(torch.float32)
    model = NRCMA(nrcma_config, glove_embeddings)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=config['t']['learning_rate'])

    if retrain:
        ckpt = torch.load(local_ckpt_path)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    loss_fn = nn.MSELoss()

    summary(model)

    train_dataloader, val_dataloader, test_dataloader = final_dataset.train_dataloader, final_dataset.val_dataloader, final_dataset.test_dataloader
    train(model, run)

    test_loss = evaluate(model, test_dataloader)
    run.summary['test/test_loss'] = test_loss
    print(f'Final test loss - {test_loss}')

    os.makedirs(save_dir, exist_ok=True)
    checkpoint_path = os.path.join(save_dir, f"{wandb_model}_checkpoint_final.pt")

    torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'model_config': config,
        }, checkpoint_path)

    run.log_model(f"./checkpoints/{wandb_model}_checkpoint_final.pt", name=wandb_model)
    run.finish()