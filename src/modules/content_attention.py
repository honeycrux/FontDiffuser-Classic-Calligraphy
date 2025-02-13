import torch
import torch.nn as nn
import torch.nn.functional as F

class RelativePositionBias(nn.Module):
    def __init__(self, heads, max_pos=64):
        super().__init__()
        self.heads = heads
        self.max_pos = max_pos
        self.pos_table = nn.Parameter(torch.randn(2*max_pos-1, heads))

    def forward(self, query_len, key_len):
        pos = torch.arange(query_len)[:, None] - torch.arange(key_len)[None, :]
        pos = pos.clamp(-self.max_pos+1, self.max_pos-1) + self.max_pos -1
        return self.pos_table[pos].permute(2, 0, 1)  # Shape: [heads, Q, K]

def window_partition(x, window_size):
    B, L, C = x.shape
    x = x.view(B, L // window_size, window_size, C)
    return x.permute(0, 2, 1, 3)  # [B, num_windows, window_size, C]

# ...existing code...

class MultiHeadContentAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(MultiHeadContentAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (
            self.head_dim * heads == embed_size
        ), "Embedding size needs to be divisible by heads"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)
        self.pos_bias = RelativePositionBias(heads)  

    def forward(self, values, keys, query, mask=None):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        # Split the embedding into self.heads different pieces
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        # Check shapes before window partition
        print(f"Values shape before window partition: {values.shape}")
        print(f"Keys shape before window partition: {keys.shape}")
        print(f"Queries shape before window partition: {queries.shape}")

        # Reshape to (N * heads, L, head_dim) for window partition
        values = values.permute(0, 2, 1, 3).reshape(N * self.heads, value_len, self.head_dim)
        keys = keys.permute(0, 2, 1, 3).reshape(N * self.heads, key_len, self.head_dim)
        queries = queries.permute(0, 2, 1, 3).reshape(N * self.heads, query_len, self.head_dim)

        # Window-based attention
        window_size = 8  # assuming window size of 8
        values_win = window_partition(values, window_size)
        keys_win = window_partition(keys, window_size)
        queries_win = window_partition(queries, window_size)

        # Check shapes after window partition
        print(f"Values shape after window partition: {values_win.shape}")
        print(f"Keys shape after window partition: {keys_win.shape}")
        print(f"Queries shape after window partition: {queries_win.shape}")

        # Scaled dot-product attention
        energy = torch.einsum("bnqd,bnkd->bnqk", [queries_win, keys_win])
        
        # Adjust the shape of pos_bias to match the energy tensor
        pos_bias = self.pos_bias(queries_win.shape[2], keys_win.shape[2]).unsqueeze(0)
        pos_bias = pos_bias.expand(energy.shape[0], -1, -1, -1)
        energy += pos_bias  # Add relative position bias

        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)

        out = torch.einsum("bnql,bnld->bnqd", [attention, values_win]).reshape(
            N, query_len, self.heads * self.head_dim
        )

        out = self.fc_out(out)
        return out

# ...existing code...
    
class ContentFeedForward(nn.Module):
    def __init__(self, embed_size, ff_hidden_dim):
        super(ContentFeedForward, self).__init__()
        self.fc1 = nn.Linear(embed_size, ff_hidden_dim)
        self.fc2 = nn.Linear(ff_hidden_dim, embed_size)
        self.dropout = nn.Dropout(0.1)  # prevent overfitting

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)  
        x = self.fc2(x)
        return x

class RMSNorm(nn.Module):
    def __init__(self, embed_size, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(embed_size))

    def forward(self, x):
        norm = x.norm(2, dim=-1, keepdim=True)
        return self.scale * x / (norm + self.eps)

class ContentAttentionModel(nn.Module):
    def __init__(self, embed_size, heads, ff_hidden_dim):
        super(ContentAttentionModel, self).__init__()
        self.attention = MultiHeadContentAttention(embed_size, heads)
        self.feed_forward = ContentFeedForward(embed_size, ff_hidden_dim)
        self.norm1 = RMSNorm(embed_size)  
        self.norm2 = RMSNorm(embed_size)
        self.alpha = nn.Parameter(torch.tensor(0.1))  
        self.dropout = nn.Dropout(0.1)  

    def forward(self, x):
        attn_out = self.attention(x, x, x, mask=None)
        x = self.norm1(self.alpha * attn_out + x) 
        x = self.dropout(x)  
        ff_out = self.feed_forward(x)
        out = self.norm2(self.alpha * ff_out + x)  
        return out

# Example usage
if __name__ == "__main__":
    embed_size = 1024
    heads = 8
    ff_hidden_dim = 2048
    C = 64  

    content_tensors = torch.randn(16, 64, 48, 48)  

    # Tokenization
    content_tensors = content_tensors.view(16, C, -1).permute(0, 2, 1).reshape(-1, embed_size)

    model = ContentAttentionModel(embed_size, heads, ff_hidden_dim)
    print("Model initialized.")

    outputs = model(content_tensors.unsqueeze(0))

   
    final_content_tensor = outputs.view(16, 48 * 48, C).permute(0, 2, 1).view(16, C, 48, 48)
    print("Inference complete.")
    print("Final Content Tensor Shape:", final_content_tensor.shape)
    print("Final Content Tensor:\n", final_content_tensor)