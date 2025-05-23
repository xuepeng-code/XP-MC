import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossModalAttention(nn.Module):
    def __init__(self, feature_dim, shared_fc=None):
        super(CrossModalAttention, self).__init__()
        self.query_proj = nn.Linear(feature_dim, feature_dim)
        self.key_proj = nn.Linear(feature_dim, feature_dim)
        self.value_proj = nn.Linear(feature_dim, feature_dim)
        if shared_fc is not None:
            self.FC = shared_fc
        else:
            self.FC = nn.Linear(256, 128)
    def forward(self, query, key):
        """
        query: Tensor of shape (batch_size, feature_dim)
        key: Tensor of shape (batch_size, feature_dim)
        """
        # 投影
        Q = self.query_proj(query)  # (batch_size, feature_dim)
        K = self.key_proj(key)      # (batch_size, feature_dim)
        V = self.value_proj(key)    # (batch_size, feature_dim)
        attention_scores = F.cosine_similarity(Q, K, dim=-1)  # (batch_size,)
        attention_weights = attention_scores.unsqueeze(-1)    # (batch_size, 1)
        attention_weights = (attention_weights + 1) / 2  # [-1, 1] to [0, 1]
        attended1 = attention_weights * V  # (batch_size, feature_dim)
        attended2 = attention_weights * Q
        combine = torch.cat([attended1,attended2],dim=1)
        attended = self.FC(combine)
        return attended, attention_weights  





class SelfAttentionMLP(nn.Module):
    def __init__(self, embed_size):
        super(SelfAttentionMLP, self).__init__()
        self.embed_size = embed_size
        self.query_linear = nn.Linear(embed_size, embed_size)
        self.key_linear = nn.Linear(embed_size, embed_size)
        self.value_linear = nn.Linear(embed_size, embed_size)
        self.fc_out = nn.Linear(embed_size, embed_size)

    def forward(self,x, mask=None):
        # Step 1: Linear transformation of values, keys, and queries
        Q = self.query_linear(x).unsqueeze(2)  # (batch_size, seq_len, embed_size)
        K = self.key_linear(x).unsqueeze(2)      # (batch_size, seq_len, embed_size)
        V = self.value_linear(x).unsqueeze(2)  # (batch_size, seq_len, embed_size)


        # Step 2: Calculate attention scores (Q @ K.T)
        energy = torch.bmm(Q, K.transpose(1, 2))  # (batch_size, seq_len, seq_len)

        if mask is not None:
            energy = energy.masked_fill(mask == 0, float('-inf'))  # Apply mask if necessary

        attention = torch.softmax(energy, dim=-1)  # (batch_size, seq_len, seq_len)
        # print(attention.shape)
        attention_weights = attention.mean(dim=-1)
        # Step 3: Weighted sum of values using attention scores
        out = torch.bmm(attention, V)  # (batch_size, seq_len, embed_size)
        out = out.view(out.size(0), -1)
        # Step 4: Final linear transformation
        out = self.fc_out(out)  # (batch_size, seq_len, embed_size)

        return attention_weights,out

