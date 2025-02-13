import torch
import torch.nn as nn
import torch.nn.functional as F
from style_attention import StyleAttentionModel
from content_attention import ContentAttentionModel

style_feature_extractor_input_singleton = (1024, 3, 3)
content_feature_extractor_1_input_singleton = (3, 96, 96)
content_feature_extractor_2_input_singleton = (64, 48, 48)
content_feature_extractor_3_input_singleton = (128, 24, 24)
content_feature_extractor_4_input_singleton = (256, 12, 12)
content_feature_extractor_5_input_singleton = (256, 12, 12)

# class KFeatureExtractorUnit_Conv(nn.Module):
#     def __init__(self, K, singleton_shape):
#         super(KFeatureExtractorUnit_Conv, self).__init__()

#         # self.K = K
#         # self.singleton_shape = singleton_shape
#         # self.singleton_size = int(torch.prod(torch.tensor(self.singleton_shape)))

#         self.conv1 = nn.Conv3d(in_channels=K, out_channels=64, kernel_size=3, padding=1)
#         self.act1 = nn.ReLU()

#         self.conv2 = nn.Conv3d(in_channels=64, out_channels=32, kernel_size=3, padding=1)
#         self.act2 = nn.ReLU()

#         self.conv3 = nn.Conv3d(in_channels=32, out_channels=1, kernel_size=3, padding=1)

#     def forward(self, style_features: torch.Tensor):
#         x = style_features
#         x = self.conv1(x)
#         x = self.act1(x)
#         x = self.conv2(x)
#         x = self.act2(x)
#         x = self.conv3(x)
#         x = x.squeeze(dim=1) # (B, 1, C, H, W) -> (B, C, H, W)
#         return x

# class KFeatureExtractorUnit_FC(nn.Module):
#     def __init__(self, K, singleton_shape):
#         super(KFeatureExtractorUnit_FC, self).__init__()

#         # self.K = K
#         # self.singleton_shape = singleton_shape
#         # self.singleton_size = int(torch.prod(torch.tensor(self.singleton_shape)))

#         self.fc1 = nn.Linear(in_features=K, out_features=4*K)
#         self.act1 = nn.ReLU()

#         self.fc2 = nn.Linear(in_features=4*K, out_features=1)

#     def forward(self, style_features: torch.Tensor):
#         # expected input shape: (BATCH, K, C, H, W)

#         dims = range(len(style_features.shape))
#         permute_in = (dims[0], *dims[2:], 1) # (B, K, C, H, W) -> (B, C, H, W, K)

#         x = style_features.permute(permute_in)
#         x = self.fc1(x)
#         x = self.act1(x)
#         x = self.fc2(x)
#         x = x.squeeze(dim=-1) # (B, C, H, W, 1) -> (B, C, H, W)

#         return x

# class KFeatureExtractorUnit_X1(nn.Module):
#     def __init__(self, K, singleton_shape):
#         super(KFeatureExtractorUnit_X1, self).__init__()

#         # self.K = K
#         # self.singleton_shape = singleton_shape
#         # self.singleton_size = int(torch.prod(torch.tensor(self.singleton_shape)))

#         self.conv1 = nn.Conv3d(in_channels=K, out_channels=32, kernel_size=3, padding=1)
#         self.act1 = nn.ReLU()

#         self.conv2 = nn.Conv3d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
#         self.act2 = nn.ReLU()

#         self.fc3 = nn.Linear(in_features=64, out_features=128)
#         self.act3 = nn.ReLU()

#         self.fc4 = nn.Linear(in_features=128, out_features=128)
#         self.act4 = nn.ReLU()

#         self.fc5 = nn.Linear(in_features=128, out_features=64)
#         self.act5 = nn.ReLU()

#         self.conv6 = nn.Conv3d(in_channels=64, out_channels=32, kernel_size=3, padding=1)
#         self.act6 = nn.ReLU()

#         self.conv7 = nn.Conv3d(in_channels=32, out_channels=1, kernel_size=3, padding=1)
#         self.act7 = nn.ReLU()

#     def forward(self, style_features: torch.Tensor):
#         permute_in = (0, 2, 3, 4, 1) # (B, K, C, H, W) -> (B, C, H, W, K)
#         permute_out = (0, 4, 1, 2, 3) # (B, C, H, W, K) -> (B, K, C, H, W)

#         x = style_features
#         x = self.conv1(x)
#         x = self.act1(x)
#         x = self.conv2(x)
#         x = self.act2(x)
#         x = x.permute(permute_in)
#         x = self.fc3(x)
#         x = self.act3(x)
#         x = self.fc4(x)
#         x = self.act4(x)
#         x = self.fc5(x)
#         x = self.act5(x)
#         x = x.permute(permute_out)
#         x = self.conv6(x)
#         x = self.act6(x)
#         x = self.conv7(x)
#         x = self.act7(x)
#         x = x.squeeze(dim=1) # (B, 1, C, H, W) -> (B, C, H, W)
#         return x

#Date: 12/2, try to apply attention model to KFeatureExtractor
class KFeatureExtractor(nn.Module):
    def __init__(self, K):
        super(KFeatureExtractor, self).__init__()
        # self.style_feature_extractor = KFeatureExtractorUnit_X1(K=K, singleton_shape=style_feature_extractor_input_singleton)
        # self.content_feature_extractor1 = KFeatureExtractorUnit_X1(K=K, singleton_shape=content_feature_extractor_1_input_singleton)
        # self.content_feature_extractor2 = KFeatureExtractorUnit_X1(K=K, singleton_shape=content_feature_extractor_2_input_singleton)
        # self.content_feature_extractor3 = KFeatureExtractorUnit_X1(K=K, singleton_shape=content_feature_extractor_3_input_singleton)
        # self.content_feature_extractor4 = KFeatureExtractorUnit_X1(K=K, singleton_shape=content_feature_extractor_4_input_singleton)
        # self.content_feature_extractor5 = KFeatureExtractorUnit_X1(K=K, singleton_shape=content_feature_extractor_5_input_singleton)
        self.attention_model = StyleAttentionModel(embed_size, heads, ff_hidden_dim)
        # to do: 5 attention models for content feature extractors
        self.content_attention_models = nn.ModuleList([
            ContentAttentionModel(embed_size, heads, ff_hidden_dim),
            ContentAttentionModel(embed_size, heads, ff_hidden_dim),
            ContentAttentionModel(embed_size, heads, ff_hidden_dim),
            ContentAttentionModel(embed_size, heads, ff_hidden_dim),
            ContentAttentionModel(embed_size, heads, ff_hidden_dim)
        ])
        

    def forward(self, style_features: torch.Tensor, content_features: list[torch.Tensor]):

        batch_size = style_features.size(0)
        K = style_features.size(1)

        #tokenization and attention for style features
        style_features = style_features.view(batch_size, K, 1024, -1).permute(0, 1, 3, 2).reshape(batch_size, -1, 1024)  #(batch_size, 9K, 1024)
        adjusted_style_features = self.attention_model(style_features)  # (batch_size, 9K, 1024)
        # return to (batch_size, k, 1024, 3, 3)
        adjusted_style_features = adjusted_style_features.view(batch_size, 9, K, 1024).permute(0, 2, 3, 1).view(batch_size, K, 1024, 3, 3) # (batch_size, 3, 3, 1024)
        # take average
        style_features = adjusted_style_features.mean(dim=1)  # (batch_size, 1024, 3, 3)

# Tokenization and attention for content features
        adjusted_content_features = []
        for i, content_feature in enumerate(content_features):
            C, H, W = content_feature.shape[2:]
            content_feature = content_feature.view(batch_size, K, C, -1).permute(0, 1, 3, 2).reshape(batch_size, -1, C)  # (batch_size, HWK, C)
            adjusted_content_feature = self.content_attention_models[i](content_feature)  # (batch_size, HWK, C)
            adjusted_content_feature = adjusted_content_feature.view(batch_size, H * W, K, C).permute(0, 2, 3, 1).view(batch_size, K, C, H, W)  # (batch_size, K, C, H, W)
            adjusted_content_features.append(adjusted_content_feature.mean(dim=1))  # (batch_size, C, H, W)

        return style_features, content_features

def report_parameters_count(name, model):
    count = 0
    for param in model.parameters():
        count += torch.prod(torch.tensor(param.size()))
    print(f"{name} Parameters: {count}")

if __name__ == '__main__':
    test_batch_num = 16
    test_k = 5
    embed_size = 1024
    heads = 8
    ff_hidden_dim = 2048

    model = KFeatureExtractor(test_k, embed_size, heads, ff_hidden_dim)
    report_parameters_count("StyleFeatureExtractor", model.attention_model)
    for i in range(5):
        report_parameters_count(f"ContentFeatureExtractor{i+1}", model.content_attention_models[i])
    report_parameters_count("KFeatureExtractor", model)

    style_features = torch.randn(test_batch_num, test_k, * style_feature_extractor_input_singleton)
    content_features = [
        torch.randn(test_batch_num, test_k, * content_feature_extractor_1_input_singleton),
        torch.randn(test_batch_num, test_k, * content_feature_extractor_2_input_singleton),
        torch.randn(test_batch_num, test_k, * content_feature_extractor_3_input_singleton),
        torch.randn(test_batch_num, test_k, * content_feature_extractor_4_input_singleton),
        torch.randn(test_batch_num, test_k, * content_feature_extractor_5_input_singleton),
    ]

    style_features_final, content_features_final = model(style_features, content_features)
    print(f"Style Features Shape: {style_features_final.shape}")
    for i, content_feature in enumerate(content_features_final):
        print(f"Content Features Shape ({i}): {content_feature.shape}")
