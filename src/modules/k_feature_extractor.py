import torch
import torch.nn as nn
import torch.nn.functional as F
from src.modules.style_attention import StyleAttentionModel
from src.modules.content_attention import ContentAttentionModel

style_feature_extractor_input_singleton = (1024, 3, 3)
content_feature_extractor_1_input_singleton = (3, 96, 96)
content_feature_extractor_2_input_singleton = (64, 48, 48)
content_feature_extractor_3_input_singleton = (128, 24, 24)
content_feature_extractor_4_input_singleton = (256, 12, 12)
content_feature_extractor_5_input_singleton = (256, 12, 12)


class KFeatureExtractor(nn.Module):
    def __init__(self, K, embed_size, heads, ff_hidden_dim):
        super(KFeatureExtractor, self).__init__()
        self.attention_model = StyleAttentionModel(embed_size, heads, ff_hidden_dim)
        self.content_attention_models = nn.ModuleList(
            [
                ContentAttentionModel(embed_size, heads, ff_hidden_dim),
                ContentAttentionModel(embed_size, heads, ff_hidden_dim),
                ContentAttentionModel(embed_size, heads, ff_hidden_dim),
                ContentAttentionModel(embed_size, heads, ff_hidden_dim),
                ContentAttentionModel(embed_size, heads, ff_hidden_dim),
            ]
        )

    def forward(
        self, style_features: torch.Tensor, content_features: list[torch.Tensor]
    ):
        # Tokenization and attention for style features
        adjusted_style_features = self.attention_model(
            style_features
        )  # (batch_size, 1024, 3, 3)
        style_features = adjusted_style_features.mean(dim=1)  # (batch_size, 1024, 3, 3)

        # Tokenization and attention for content features
        adjusted_content_features = []
        for i, content_feature in enumerate(content_features):
            adjusted_content_feature = self.content_attention_models[i](
                content_feature
            )  # (batch_size, C, H, W)
            adjusted_content_features.append(
                adjusted_content_feature.mean(dim=1)
            )  # (batch_size, C, H, W)

        return style_features, adjusted_content_features


def report_parameters_count(name, model):
    count = 0
    for param in model.parameters():
        count += torch.prod(torch.tensor(param.size()))
    print(f"{name} Parameters: {count}")


if __name__ == "__main__":
    test_batch_num = 16
    test_k = 5
    embed_size = 1024
    heads = 8
    ff_hidden_dim = 2048

    model = KFeatureExtractor(test_k, embed_size, heads, ff_hidden_dim)
    report_parameters_count("StyleFeatureExtractor", model.attention_model)
    for i in range(5):
        report_parameters_count(
            f"ContentFeatureExtractor{i+1}", model.content_attention_models[i]
        )
    report_parameters_count("KFeatureExtractor", model)

    style_features = torch.randn(
        test_batch_num, test_k, *style_feature_extractor_input_singleton
    )
    content_features = [
        torch.randn(
            test_batch_num, test_k, *content_feature_extractor_1_input_singleton
        ),
        torch.randn(
            test_batch_num, test_k, *content_feature_extractor_2_input_singleton
        ),
        torch.randn(
            test_batch_num, test_k, *content_feature_extractor_3_input_singleton
        ),
        torch.randn(
            test_batch_num, test_k, *content_feature_extractor_4_input_singleton
        ),
        torch.randn(
            test_batch_num, test_k, *content_feature_extractor_5_input_singleton
        ),
    ]

    style_features_final, content_features_final = model(
        style_features, content_features
    )
    print(f"Style Features Shape: {style_features_final.shape}")
    for i, content_feature in enumerate(content_features_final):
        print(f"Content Features Shape ({i}): {content_feature.shape}")
