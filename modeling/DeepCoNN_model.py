import torch
import torch.nn as nn

# DeepCoNN Architecture
# Convolution Max-Pooling Layer
class ConvolutionMaxPooling(torch.nn.Module):
    '''
    User and Item Review Layers
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

            # No need to use the MaxPool1D as Adaptive pooling removes the size mismatch error automatically by flattening
            self.maxs.append(torch.nn.AdaptiveMaxPool1d(output_size=1))

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

        # Paper recommended ReLU, however, I wanted to experiment with others as well.
        self.activation = torch.nn.GELU() # Softer than Relu, introduced in GPT-2 transformer
        # self.activation = torch.nn.ReLU()       # Shared activation function for user and item
        self.full_connect = torch.nn.Linear(self.n1 * len(self.t), self.latent_factors)     # Shared fully connected layer

    
    def forward(self, review):
        """
        Input Shape: (Batch Size, Review Length, Word Embedding Size)
        Output Shape: (Batch Size, Latent Factors Size)
        """
        output = []
        review = review.permute(0,2,1)

        # output_u = []
        # output_i = []
        # review_u = review_u.permute(0,2,1)

        for max_pool, conv in zip(self.maxs, self.convs):
            out = self.activation(conv(review))
            max_out = max_pool(out)
            flatten_out = torch.flatten(max_out, start_dim=1)        # Passed stride=1 as an argument and it returned an error
            output.append(flatten_out)


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
        
        conv_out = torch.cat(output, dim=1)
        latent = self.full_connect(conv_out)

        return latent
    

class FactorMachine(torch.nn.Module):
    """
    Factorization Machine Paper Formula
    y^ = w0 + Sum_i w_i*x_i + sum_i=1^n sum_j=i+1^n v_i*v_j x_i*x_j
    where x is the input feature vector, w is the linear weights, v_i are the latent vectors
    Reference: https://www.kaggle.com/gennadylaptev/factorization-machine-implemented-in-pytorch
    Input Shape: (Batch Size, Latent Factors Size * 2)
    Output Shape: (Batch Size)
    """

    def __init__(self, latent_factors, fm_k):
        super().__init__()
        self.latent_factors = latent_factors
        self.fm_k = fm_k
        self.std_norm = torch.randn(self.latent_factors * 2, self.fm_k)  # Start with random values from N(0,1) for latent factor matrix
        self.V = torch.nn.Parameter(self.std_norm)
        self.lin = torch.nn.Linear(self.latent_factors * 2, 1) # w0 + Sum_i w_i*x_i i.e. linear layer

    def forward(self, x):
        s1_square = torch.matmul(x, self.V).pow(2).sum(1, keepdim=True)
        s2 = torch.matmul(x.pow(2), self.V.pow(2)).sum(1, keepdim=True)
        # Above two equations sum_i=1^n sum_j=i+1^n v_i*v_j x_i*x_j

        out_inter = 0.5 * (s1_square-s2)
        out_lin = self.lin(x)       # Got matmul error (32x1 and 100x1) so changed out_inter to x
        out = out_inter + out_lin
        return out
    

class DeepCoNN(nn.Module):
    """
    Reference: https://github.com/KindRoach/DeepCoNN-Pytorch/blob/master/model/ConvMaxLayer.svg
    https://github.com/KindRoach/DeepCoNN-Pytorch/blob/master/model/DeepCoNN.py
    """
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

        self.user_layer = ConvolutionMaxPooling(max_review_length_u, t, embed_dim, n1, latent_factors)
        self.item_layer = ConvolutionMaxPooling(max_review_length_i, t, embed_dim, n1, latent_factors)
        self.share_layer = FactorMachine(latent_factors, fm_k)

    def forward(self, user_review, item_review):
        """
        Input Shape: (Batch Size, Review Length)
        Output Shape: (Batch Size)
        """

        user_latent = self.user_layer(self.embedding(user_review))
        item_latent = self.item_layer(self.embedding(item_review))
        latent = torch.cat([user_latent, item_latent], dim=1)
        predict = self.share_layer(latent)
        predict = predict.squeeze(1)    # Final 1D vectors from the paper diagram, further used for dot product
        return predict