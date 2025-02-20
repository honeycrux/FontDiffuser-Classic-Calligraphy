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
        return self.pos_table[pos].permute(1, 0, 2)  # Shape: [heads, Q, K]

def window_partition(x, window_size):
    B, L, C = x.shape
    pad_len = (window_size - (L % window_size)) % window_size
    x = F.pad(x, (0, 0, 0, pad_len))
    L = L + pad_len
    x = x.view(B, L // window_size, window_size, C)
    return x.permute(0, 2, 1, 3), pad_len  # [B, window_size, num_windows, C], pad_len

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
        # print(f"Values shape before window partition: {values.shape}")
        # print(f"Keys shape before window partition: {keys.shape}")
        # print(f"Queries shape before window partition: {queries.shape}")

        # Reshape to (N * heads, L, head_dim) for window partition
        values = values.permute(0, 2, 1, 3).reshape(N * self.heads, value_len, self.head_dim)
        keys = keys.permute(0, 2, 1, 3).reshape(N * self.heads, key_len, self.head_dim)
        queries = queries.permute(0, 2, 1, 3).reshape(N * self.heads, query_len, self.head_dim)

        # Window-based attention
        window_size = min(value_len, key_len, query_len) // self.heads
        if window_size == 0:
            window_size = 1  # Ensure window_size is at least 1
        values_win, pad_len = window_partition(values, window_size)
        keys_win, _ = window_partition(keys, window_size)
        queries_win, _ = window_partition(queries, window_size)

        # Check shapes after window partition
        # print(f"Values shape after window partition: {values_win.shape}")
        # print(f"Keys shape after window partition: {keys_win.shape}")
        # print(f"Queries shape after window partition: {queries_win.shape}")

        # Scaled dot-product attention
        energy = torch.einsum("bnqd,bnkd->bnqk", [queries_win, keys_win])

        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)

        out = torch.einsum("bnql,bnld->bnqd", [attention, values_win]).reshape(
            N, query_len + pad_len, self.heads * self.head_dim
        )

        out = self.fc_out(out)
        return out[:, :query_len, :]  # Remove padding
    
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
        # Tokenization
        B, K, C, H, W = x.shape
        # print("Content Tensor Shape:", x.shape)
        L = (K * C * H * W) // 1024
        x = x.view(B, L, 1024)
        # print("Tokenization Content Tensor Shape:", x.shape)

        # Multi-head attention
        attn_out = self.attention(x, x, x, mask=None)
        
        # RMS
        x = self.norm1(self.alpha * attn_out + x)
        
        # Dropout
        x = self.dropout(x)
        
        # Feed forward
        ff_out = self.feed_forward(x)
        
        # RMS
        out = self.norm2(self.alpha * ff_out + x)

        # Reshape
        out = out.view(B, K, H * W, C).permute(0, 1, 3, 2).view(B, K, C, H, W)

        # print("Inference complete.")
        # print("Final Content Tensor Shape:", out.shape)
        
        return out

# Example usage
if __name__ == "__main__":
    embed_size = 1024
    heads = 8
    ff_hidden_dim = 2048
    batch_size = 32  

    content_tensors_list = [
        torch.randn(batch_size, 5, 3, 96, 96),  
        torch.randn(batch_size, 5, 64, 48, 48),  
        torch.randn(batch_size, 5, 128, 24, 24),  
        torch.randn(batch_size, 5, 256, 12, 12), 
    ]

    model = ContentAttentionModel(embed_size, heads, ff_hidden_dim)
    print("Model initialized.")

    for content_tensors in content_tensors_list:
        B, K, C, H, W = content_tensors.shape

        outputs = model(content_tensors)

        print("Final Content Tensor:\n", outputs)
