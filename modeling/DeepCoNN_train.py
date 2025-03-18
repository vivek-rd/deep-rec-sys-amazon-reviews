import os
import time
import random
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import wandb

from torchinfo import summary
from modeling.DeepCoNN_model import DeepCoNN
from utils import final_dataset_DeepCoNN

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

            optimizer.zero_grad(set_to_none=True)
            user, item, user_tower_input, item_tower_input, true_rating = batch
            # u_embed, i_embed, rating = u_embed.to(device), i_embed.to(device), rating.to(device)
            user_tower_input, item_tower_input, true_rating, user, item = user_tower_input.to(device), \
            item_tower_input.to(device), true_rating.to(device), user.to(device), item.to(device) 
            # predicted_rating = model(user_tower_input, item_tower_input, user, item)
            predicted_rating = model(user_tower_input, item_tower_input)
            loss = loss_fn(predicted_rating, true_rating)
            metrics = {'train/train_loss': loss.item()}

            loss.backward()
            optimizer.step()

            if index != len(train_dataloader) - 1:
                run.log(metrics)
                print(f'Epoch - {i} | step - {index} | train loss - {loss:.4f}')

            # if batches_trained % eval_iters == 0 and index != len(train_dataloader) - 1 and batches_trained != 0 :
            #     val_loss = evaluate(model, val_dataloader)
            #     metrics['val/val_loss'] = val_loss
            #     print(f'Epoch - {i} | step - {index} | train loss - {loss:.4f} | val loss - {val_loss:.4f}')
            #     run.log(metrics)
            
            batches_trained += 1

        val_loss = evaluate(model, val_dataloader)
        metrics['val/val_loss'] = val_loss
        print(f'Epoch - {i} | step - {index} | train loss - {loss:.4f} | val loss - {val_loss:.4f}')
        run.log(metrics)


@torch.no_grad()
def evaluate(model, dataloader):
    model.eval()
    total_error = 0.0
    batches_evaluated = 0
    
    for index, batch in enumerate(dataloader):
        user, item, user_tower_input, item_tower_input, true_rating = batch
        user_tower_input, item_tower_input, true_rating, user, item = user_tower_input.to(device), \
            item_tower_input.to(device), true_rating.to(device), user.to(device), item.to(device) 
        # predicted_rating = model(user_tower_input, item_tower_input, user, item)
        predicted_rating = model(user_tower_input, item_tower_input)
        error = loss_fn(predicted_rating, true_rating)
        total_error += error.item()
        
        batches_evaluated += 1
    
    avg_error = total_error / batches_evaluated
    return avg_error


if __name__ == '__main__':
    wandb_entity = "dhopate-r-northeastern-university"
    wandb_project = "ds_capstone_recsys"
    wandb_run_name = "demo_run_1"
    wandb_tags = ["deepconn", wandb_run_name, "retrain"]
    wandb_model = wandb_run_name + "_model"
    wandb_model_version = "v0"

    epochs = 5
    max_review_length_u = 200
    max_review_length_i = 1000
    embed_dim = 300
    t = [3, 5]                  # Kernel Width
    n1 = 100                    # Kernel Depth
    latent_factors = 50 
    fm_k = 10           # Number of factors in Factorization Machine
    learning_rate = 1e-4
    reg_lambda = 1e-3   # Regularization Lambda
    batch_size = 512
    config={"epochs":epochs, "max_review_length_u":max_review_length_u, "max_review_length_i":max_review_length_i,
            "embed_dim":embed_dim, "t":t, "n1":n1, "latent_factors":latent_factors, "fm_k":fm_k,
             "learning_rate":learning_rate, "reg_lambda":reg_lambda}
    
    api = wandb.Api()
    save_dir = './checkpoints'

    run = wandb.init(project=wandb_project, 
                config=config, 
                tags=wandb_tags,
                name=wandb_run_name + '_' + time.ctime())

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")  # macOS Metal Performance Shaders
    else:
        device = torch.device("cpu")
    
    print(f"Device: {device}")

    seed_everything(42)
    glove_embeddings = torch.load('data/required_embeddings.pt').to(torch.float32)
    model = DeepCoNN(max_review_length_u=max_review_length_u, max_review_length_i=max_review_length_i, t=t, embed_dim=embed_dim, 
                     n1=n1, latent_factors=latent_factors, fm_k=fm_k, glove_embeddings=glove_embeddings)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=reg_lambda)

    loss_fn = nn.MSELoss()
    summary(model)

    train_dataloader, val_dataloader, test_dataloader = final_dataset_DeepCoNN.train_dataloader, final_dataset_DeepCoNN.val_dataloader, final_dataset_DeepCoNN.test_dataloader
    train(model, run)

    test_loss = evaluate(model, test_dataloader)
    run.summary['test/test_loss'] = test_loss
    print(f'Final test loss - {test_loss}')

    os.makedirs(save_dir, exist_ok=True)
    checkpoint_path = os.path.join(save_dir, f"{wandb_model}_checkpoint_final.pt")

    torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'config':config
        }, checkpoint_path)

    run.log_model(f"./checkpoints/{wandb_model}_checkpoint_final.pt", name=wandb_model)
    run.finish()
