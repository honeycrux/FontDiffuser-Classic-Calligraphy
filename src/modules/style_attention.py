import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttention(nn.Module):
    def __init__(self, embed_size, reduction_ratio=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(embed_size, embed_size // reduction_ratio),
            nn.ReLU(),
            nn.Linear(embed_size // reduction_ratio, embed_size)
        )
    def forward(self, x):
        b, _, _ = x.size()
        y = self.avg_pool(x.transpose(1,2)).view(b, -1)
        y = self.fc(y).view(b, 1, -1)  # [batch_size, 1, embed_size]
        return x * y.sigmoid()

class MultiHeadStyleAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(MultiHeadStyleAttention, self).__init__()
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
        self.channel_attn = ChannelAttention(embed_size) 

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

        # Scaled dot-product attention
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )

        out = self.fc_out(out)
        out = self.channel_attn(out) 
        return out

class AdaIN(nn.Module):
    def __init__(self, embed_size):
        super().__init__()
        self.embed_size = embed_size

    def forward(self, x, style_mean, style_std):
        x_mean = x.mean(dim=1, keepdim=True)
        x_std = x.std(dim=1, keepdim=True)
        return style_std * (x - x_mean) / (x_std + 1e-5) + style_mean

class StyleFeedForward(nn.Module):
    def __init__(self, embed_size, ff_hidden_dim):
        super(StyleFeedForward, self).__init__()
        self.fc1 = nn.Linear(embed_size, ff_hidden_dim)
        self.fc2 = nn.Linear(ff_hidden_dim, embed_size)
        self.fc3 = nn.Linear(embed_size, embed_size)  
        self.dropout = nn.Dropout(0.1)  
        self.adain = AdaIN(embed_size)  

    def forward(self, x):
        style_mean = x.mean(dim=1, keepdim=True)
        style_std = x.std(dim=1, keepdim=True)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)  
        x = self.adain(x, style_mean, style_std)  
        x = self.dropout(x)
        x = self.fc3(x)
        return x

class StyleAttentionModel(nn.Module):
    def __init__(self, embed_size, heads, ff_hidden_dim):
        super(StyleAttentionModel, self).__init__()
        self.attention = MultiHeadStyleAttention(embed_size, heads)
        self.feed_forward = StyleFeedForward(embed_size, ff_hidden_dim)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.dropout = nn.Dropout(0.1)  

    def forward(self, x):
        attn_out = self.attention(x, x, x, mask=None)
        x = self.norm1(attn_out + x)
        x = self.dropout(x)  
        ff_out = self.feed_forward(x)
        out = self.norm2(ff_out + x)
        return out

# Example usage
if __name__ == "__main__":
    embed_size = 1024
    heads = 8
    ff_hidden_dim = 2048
    K = 9  

    style_tensors = torch.randn(K, 3, 3, 1024)

    # Tokenization
    style_tensors = style_tensors.view(K, 1024, -1).permute(0, 2, 1).reshape(-1, 1024)

    model = StyleAttentionModel(embed_size, heads, ff_hidden_dim)
    print("Model initialized.")

    outputs = model(style_tensors.unsqueeze(0))  

    final_style_tensor = outputs.reshape(K, 9, 1024).permute(0, 2, 1).reshape(K, 3, 3, 1024)
    print("Inference complete.")
    print("Final Style Tensor Shape:", final_style_tensor.shape)
    print("Final Style Tensor:\n", final_style_tensor)