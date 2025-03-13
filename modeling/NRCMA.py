import math
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

@dataclass
class NRCMAConfig:
    num_users: int
    num_products: int
    num_conv_filters: int
    kernel_size: int
    word_embed_dim: int
    id_embed_dim: int
    attention_vector_dim: int
    feature_vector_dim: int
    words_per_sentence: int
    user_reviews_per_entity: int
    item_reviews_per_entity: int

    @staticmethod
    def from_config(config):
        return NRCMAConfig(**config)


class NRCMA(nn.Module):
    def __init__(self, config: NRCMAConfig, glove_embeddings):
        super(NRCMA, self).__init__()

        # Extract all config parameters into local variables
        self.num_users = config.num_users
        self.num_products = config.num_products
        self.id_embed_dim = config.id_embed_dim
        self.num_conv_filters = config.num_conv_filters
        self.kernel_size = config.kernel_size
        self.word_embed_dim = config.word_embed_dim
        self.attention_vector_dim = config.attention_vector_dim
        self.feature_vector_dim = config.feature_vector_dim
        
        self.word2vec = nn.Embedding.from_pretrained(glove_embeddings, freeze=True)
        self.user_embedding = nn.Embedding(self.num_users, self.id_embed_dim)
        self.item_embedding = nn.Embedding(self.num_products, self.id_embed_dim)

        # user tower
        self.user_cnn = nn.Conv1d(in_channels=self.word_embed_dim, 
                                  out_channels=self.num_conv_filters, 
                                  kernel_size=self.kernel_size, 
                                  padding='same', 
                                  bias=True)
        self.word_level_matrix = nn.Linear(self.id_embed_dim, self.attention_vector_dim)
        self.word_harmony_matrix = nn.Linear(self.num_conv_filters, self.attention_vector_dim)
        self.review_level_matrix = nn.Linear(self.id_embed_dim, self.attention_vector_dim)
        self.review_harmony_matrix = nn.Linear(self.num_conv_filters, self.attention_vector_dim)

        # item tower
        self.item_cnn = nn.Conv1d(in_channels=self.word_embed_dim, 
                                  out_channels=self.num_conv_filters, 
                                  kernel_size=self.kernel_size, 
                                  padding='same', 
                                  bias=True)
        self.item_word_level_matrix = nn.Linear(self.id_embed_dim, self.attention_vector_dim)
        self.item_word_harmony_matrix = nn.Linear(self.num_conv_filters, self.attention_vector_dim)
        self.item_review_level_matrix = nn.Linear(self.id_embed_dim, self.attention_vector_dim)
        self.item_review_harmony_matrix = nn.Linear(self.num_conv_filters, self.attention_vector_dim)

        # factorization machine, 2 * num_conv_filters as we get concat features from user, item
        self.fm_linear = nn.Linear(2*self.num_conv_filters, 1)
        self.v = nn.Parameter(torch.empty(2*self.num_conv_filters, self.feature_vector_dim))
        
        # Initialize self.v similar to nn.Linear weights
        nn.init.kaiming_uniform_(self.v, a=math.sqrt(5))


    def forward(self, user_input, item_input, user_id, item_id):
        
        user_embedding = self.user_embedding(user_id)
        item_embedding = self.item_embedding(item_id)
        
        user_input = self.word2vec(user_input.to(torch.int))
        item_input = self.word2vec(item_input.to(torch.int))

        d_u = self.process_single_tower(user_input, item_embedding, tower='user')
        d_i = self.process_single_tower(item_input, user_embedding, tower='item')
        o = torch.cat((d_u, d_i), dim=1)

        # factorization machine
        linear_part = self.fm_linear(o).squeeze(1) 
        interaction = o.unsqueeze(2) * self.v
        square_of_sum = (interaction.sum(dim=1) ** 2)  
        sum_of_square = (interaction ** 2).sum(dim=1)

        interaction_part = 0.5 * (square_of_sum - sum_of_square).sum(dim=1)
        prediction = linear_part + interaction_part
        
        return prediction

    def process_single_tower(self, input_matrix, embedding, tower):

        if tower == 'user':
            cnn = self.user_cnn
            word_level_matrix = self.word_level_matrix
            word_harmony_matrix = self.word_harmony_matrix
            review_level_matrix = self.review_level_matrix
            review_harmony_matrix = self.review_harmony_matrix
        else:
            cnn = self.item_cnn
            word_level_matrix = self.item_word_level_matrix
            word_harmony_matrix = self.item_word_harmony_matrix
            review_level_matrix = self.item_review_level_matrix
            review_harmony_matrix = self.item_review_harmony_matrix

        # process information through single tower - employing cross attention
        batch_size = input_matrix.shape[0]
        input_matrix = input_matrix.flatten(start_dim=0, end_dim=1).transpose(1, 2)
        review_features = F.relu(cnn(input_matrix))
        beta_k = F.relu(word_level_matrix(embedding))
        
        review_features = review_features.view(batch_size, -1, *review_features.shape[1:])
        
        intermediate_output_1 = word_harmony_matrix(review_features.transpose(-1, -2)).transpose(-1, -2) # not sure if this step is correct; need to check this
        beta_k = beta_k.unsqueeze(1)
        beta_k = beta_k.unsqueeze(2)
        
        b_c = beta_k @ intermediate_output_1
        b_c = b_c.squeeze(2)
        alpha_c = nn.Softmax(dim=-1)(b_c)
        
        d_uk = alpha_c.unsqueeze(2) @ review_features.transpose(-1, -2)
        d_uk = d_uk.squeeze(2)

        beta_u = F.relu(review_level_matrix(embedding))
        intermediate_output_2 = review_harmony_matrix(d_uk)
        b_k = beta_u.unsqueeze(1) @ intermediate_output_2.transpose(1, 2)

        alpha_k = nn.Softmax(dim=-1)(b_k)
        d_u = alpha_k @ d_uk
        d_u = d_u.squeeze(1)

        return d_u


if __name__=="__main__":
    
    with open('../config/nrcma.yaml') as f:
        config = yaml.safe_load(f)
    
    nrcma_config = NRCMAConfig.from_config(config['model'])
    glove_embeddings = torch.load('required_embeddings.pt')
    model = NRCMA(nrcma_config, glove_embeddings)
    print(next(model.parameters()).dtype)
    
    user_input = torch.randn(32, 64, 16, 300)
    item_input = torch.randn(32, 64, 16, 300)
    
    user_id = torch.randint(0, nrcma_config.num_users, (32,))
    item_id = torch.randint(0, nrcma_config.num_products, (32,))
    output = model(user_input, item_input, user_id, item_id)
    
    print(output.shape)