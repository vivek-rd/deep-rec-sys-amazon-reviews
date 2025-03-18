import torch
import torch.nn as nn

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
    def __init__(self, max_review_length_u, max_review_length_i, t, embed_dim, n1, latent_factors, fm_k, glove_embeddings):
        super().__init__()

        self.max_review_length_u = max_review_length_u
        self.max_review_length_i = max_review_length_i
        self.t = t
        self.embed_dim = embed_dim
        self.n1 = n1
        self.latent_factors = latent_factors
        self.fm_k = fm_k

        self.embedding = torch.nn.Embedding.from_pretrained(glove_embeddings, freeze=True)
        # self.embedding.weight.requires_grad = False

        self.user_layer = ConvMaxLayer(max_review_length_u, t, embed_dim, n1, latent_factors)
        self.item_layer = ConvMaxLayer(max_review_length_i, t, embed_dim, n1, latent_factors)
        self.share_layer = FMLayer(latent_factors, fm_k)

    def forward(self, user_review, item_review):
        """
            Input Shape: (Batch Size, Review Length)
            Output Shape: (Batch Size)
        """

        user_latent = self.user_layer(self.embedding(user_review))
        item_latent = self.item_layer(self.embedding(item_review))
        latent = torch.cat([user_latent, item_latent], dim=1)
        predict = self.share_layer(latent)
        predict = predict.squeeze(1)
        return predict