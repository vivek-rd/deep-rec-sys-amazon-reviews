import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


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
        self.input_dim = input_dim       # d
        self.hidden_dim = hidden_dim     # d_k
        self.kernel_size = kernel_size   # n
        self.num_heads = num_heads
        self.d_h = hidden_dim // num_heads  # subspace dimension d_h (assume hidden_dim divisible by num_heads)
        self.max_seq_len = max_seq_len

        # Convolution layers for computing q, k, v.
        # Using Conv1d with kernel_size = n and padding = floor((n-1)/2) gives a sliding window effect.
        # But we skip padding since our sequences are fixed-length. Simpler = better.
        padding = kernel_size // 2  # Works when n is odd.
        self.conv_q = nn.Conv1d(in_channels=input_dim, out_channels=hidden_dim, kernel_size=kernel_size, padding=padding)
        self.conv_k = nn.Conv1d(in_channels=input_dim, out_channels=hidden_dim, kernel_size=kernel_size, padding=padding)
        self.conv_v = nn.Conv1d(in_channels=input_dim, out_channels=hidden_dim, kernel_size=kernel_size, padding=padding)

        # Feed-forward layer after attention (W_f, b_f)
        self.ff = nn.Linear(hidden_dim, hidden_dim)

        # Relative positional embeddings for keys and values.
        # Assumes a fixed maximum sequence length, obviously. For each (i, j) pair, we learn a vector in R^(d_h). Fancy.
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

        # First, apply convolution. Conv1d expects (batch, channels, seq_len), so let’s flip some axes.
        x_conv = x.transpose(1, 2)       # (batch, input_dim, t)
        q = F.relu(self.conv_q(x_conv))  # (batch, hidden_dim, t)
        k = F.relu(self.conv_k(x_conv))
        v = F.relu(self.conv_v(x_conv))

        # Back to (batch, t, hidden_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Split each into H heads: reshape from (batch, t, hidden_dim) to (batch, H, t, d_h)
        q = q.view(batch, t, self.num_heads, self.d_h).transpose(1, 2)  # (batch, num_heads, t, d_h)
        k = k.view(batch, t, self.num_heads, self.d_h).transpose(1, 2)
        v = v.view(batch, t, self.num_heads, self.d_h).transpose(1, 2)

        # ---------------------------
        # Multihead Self-Attention with Relative Position Embeddings
        # ---------------------------

        # Standard attention scores: for each head h, for positions i and j: score_std[i,j] = <q_i^h, k_j^h>
        scores_std = torch.matmul(q, k.transpose(-2, -1))  # (batch, num_heads, t, t)

        # Relative position term for keys: p_k has shape (max_seq_len, max_seq_len, d_h). Slice to current sequence length.
        p_K = self.p_K[:t, :t, :]  # (t, t, d_h) PyTorch is fucking weird.

        # Compute extra scores: for each head h, each position i and j, e_relative[i,j] = <q_i^h, p_K[i,j]>
        # We do this by broadcasting: (batch, num_heads, t, 1, d_h) * (1, 1, t, t, d_h) -> sum over d_h.
        e_relative = (q.unsqueeze(3) * p_K.unsqueeze(0).unsqueeze(0)).sum(-1)  # (batch, num_heads, t, t)

        # Total attention scores(scaled)
        scores = (scores_std + e_relative) / math.sqrt(self.d_h)  # (batch, num_heads, t, t)
        attn = F.softmax(scores, dim=-1)                          # attention weights a_ij^h

        # Compute attention output.
        # First, weighted sum for v:
        z_v = torch.matmul(attn, v)                                             # (batch, num_heads, t, d_h)
        p_V = self.p_V[:t, :t, :]                                               # Now adding relative contribution from values using p_v | shape: (t, t, d_h)
        z_p = (attn.unsqueeze(-1) * p_V.unsqueeze(0).unsqueeze(0)).sum(dim=-2)  # For each i, add: sum_j a_{ij}^h * p_V^{ij} | shape: (batch, num_heads, t, d_h)
        z = z_v + z_p  # (batch, num_heads, t, d_h)

        z = z.transpose(1, 2).contiguous().view(batch, t, self.hidden_dim) # Concatenating all heads: from (batch, num_heads, t, d_h) to (batch, t, hidden_dim)
        z = F.relu(self.ff(z))                                             # feed-forward layer with ReLU (W_f and b_f in Equation (7) in paper) | shape: (batch, t, hidden_dim)

        return z
    


class SequenceAggregatingModule(nn.Module):
    def __init__(self, hidden_dim):
        """
        Args: hidden_dim: d_k (the same hidden dimension used in encoding)
        """
        super(SequenceAggregatingModule, self).__init__()
        self.hidden_dim = hidden_dim
        self.d_p = hidden_dim // 2                   # per paper, d_p = d_k/2
        self.W_p = nn.Linear(hidden_dim, self.d_p)   # This covers multiplication by W_p and addition of b_p.
        self.h = nn.Parameter(torch.randn(self.d_p)) # h is a learnable vector in R^(d_p)

    def forward(self, z):
        """
        Args:
            z: Tensor of shape (batch, seq_len, hidden_dim)
        Returns:
            l: Aggregated vector of shape (batch, hidden_dim)
        """

        r = F.relu(self.W_p(z))                       # intermediate representation: r_i = ReLU(W_p z_i + b_p) | shape: (batch, seq_len, d_p)
        scores = torch.matmul(r, self.h)              # importance scores: score_i = h^T r_i | shape: (batch, seq_len)
        attn = F.softmax(scores, dim=-1)              # Normalizing to get attention weights a_i | shape: (batch, seq_len)
        l = torch.sum(z * attn.unsqueeze(-1), dim=1)  # Weighted sum of z_i's | (batch, hidden_dim)

        return l
    

class SequenceEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, num_heads, max_seq_len):
        """
        Args:
            input_dim: Dimension of input embeddings.
            hidden_dim: Hidden dimension d_k.
            kernel_size: n-gram width.
            num_heads: Number of attention heads.
            max_seq_len: Maximum sequence length.
        """
        super(SequenceEncoder, self).__init__()
        self.encoding_module = SequenceEncodingModule(input_dim, hidden_dim, kernel_size, num_heads, max_seq_len)
        self.aggregating_module = SequenceAggregatingModule(hidden_dim)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch, seq_len, input_dim)
        Returns:
            l: Aggregated representation of shape (batch, hidden_dim)
        """
        z = self.encoding_module(x)    # (batch, seq_len, hidden_dim)
        l = self.aggregating_module(z)  # (batch, hidden_dim)
        return l
    

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

        x_sentences = x.view(B * R * S, W, E)              # Flatten reviews and sentences into one dimension: (B * R * S, W, E)
        sentence_reps = self.sentence_encoder(x_sentences) # Processing all sentences at once: each sentence → a sentence representation (B * R * S, hidden_dim)
        sentence_reps = sentence_reps.view(B, R, S, -1)    # Reshape back to (B, R, S, hidden_dim)

        x_reviews = sentence_reps.view(B * R, S, -1)       # For each review, we have S sentence representations. Flatten reviews: (B * R, S, hidden_dim)
        review_reps = self.review_encoder(x_reviews)       # Processing all reviews: (B * R, hidden_dim)
        review_reps = review_reps.view(B, R, -1)           # Reshape back to (B, R, hidden_dim)

        final_rep = self.final_encoder(review_reps)        # Processing the sequence of reviews for each entity: (B, R, hidden_dim) → (B, hidden_dim)
        return final_rep
    

class RatingPredictor(nn.Module):
    def __init__(self, rep_dim, hidden_size=32):
        super(RatingPredictor, self).__init__()
        self.fc1 = nn.Linear(rep_dim * 2, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)  # Predicts a single rating value

    def forward(self, user_rep, item_rep):
        x = torch.cat([user_rep, item_rep], dim=-1)
        x = torch.relu(self.fc1(x))
        rating = self.fc2(x)
        return rating
    

class RatingPredictor(nn.Module):
    def __init__(self, num_users, num_items, hidden_dim, latent_dim): 
        super(RatingPredictor, self).__init__()

        self.user_proj = nn.Linear(hidden_dim, latent_dim)
        self.item_proj = nn.Linear(hidden_dim, latent_dim)

        # Additional user and item embeddings the paper had, probably will improve performance
        self.user_emb = nn.Embedding(num_users, latent_dim)  # ud
        self.item_emb = nn.Embedding(num_items, latent_dim)  # id
        self.predict_layer = nn.Linear(latent_dim, 1)
        
        self.user_bias = nn.Embedding(num_users, 1)
        self.item_bias = nn.Embedding(num_items, 1)
        self.global_bias = nn.Parameter(torch.zeros(1))  # bg

    def forward(self, user_id, item_id, user_review_repr, item_review_repr):
        ur = torch.relu(self.user_proj(user_review_repr))  # u_r
        ir = torch.relu(self.item_proj(item_review_repr))  # i_r

        ud = self.user_emb(user_id)  # u_d
        id = self.item_emb(item_id)  # i_d

        u_final = ud + ur
        i_final = id + ir

        # Element-wise interaction not concatenation dumbass
        h = u_final * i_final  # ⊙ operation

        rating_pred = self.predict_layer(h).squeeze(1)  # w_f^T h
        rating_pred += self.user_bias(user_id).squeeze(1)
        rating_pred += self.item_bias(item_id).squeeze(1)
        rating_pred += self.global_bias  # b_u + b_v + b_g

        return rating_pred

    

class HSACN(nn.Module):
    def __init__(self, num_users, num_items, word_embedding_dim, hidden_dim, latent_dim, kernel_size, num_heads, words_per_sentence,
                 sentences_per_review, user_reviews_per_entity, item_reviews_per_entity, glove_embeddings):
        super(HSACN, self).__init__()
        self.user_encoder = HierarchicalEncoder(word_embedding_dim, hidden_dim, kernel_size, num_heads,
                                                  words_per_sentence, sentences_per_review, user_reviews_per_entity)
        self.item_encoder = HierarchicalEncoder(word_embedding_dim, hidden_dim, kernel_size, num_heads,
                                                  words_per_sentence, sentences_per_review, item_reviews_per_entity)
        self.rating_predictor = RatingPredictor(num_users, num_items, hidden_dim, latent_dim)

        self.word2vec = nn.Embedding.from_pretrained(glove_embeddings, freeze=True)

    def forward(self, user_id, item_id, user_input, item_input):
        user_input = self.word2vec(user_input.to(torch.int))
        item_input = self.word2vec(item_input.to(torch.int))
        
        user_rep = self.user_encoder(user_input)  # (batch, hidden_dim)
        item_rep = self.item_encoder(item_input)  # (batch, hidden_dim)
        rating = self.rating_predictor(user_id, item_id, user_rep, item_rep)  # (batch, 1)
        return rating