import torch
import torch.nn as nn

style_feature_extractor_input_singleton = (1024, 3, 3)
content_feature_extractor_1_input_singleton = (3, 96, 96)
content_feature_extractor_2_input_singleton = (64, 48, 48)
content_feature_extractor_3_input_singleton = (128, 24, 24)
content_feature_extractor_4_input_singleton = (256, 12, 12)
content_feature_extractor_5_input_singleton = (256, 12, 12)

class KFeatureExtractorUnit_Conv(nn.Module):
    def __init__(self, K, singleton_shape):
        super(KFeatureExtractorUnit_Conv, self).__init__()

        # self.K = K
        # self.singleton_shape = singleton_shape
        # self.singleton_size = int(torch.prod(torch.tensor(self.singleton_shape)))

        self.conv1 = nn.Conv3d(in_channels=K, out_channels=64, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv3d(in_channels=64, out_channels=32, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()

        self.conv3 = nn.Conv3d(in_channels=32, out_channels=1, kernel_size=3, padding=1)

    def forward(self, style_features: torch.Tensor):
        x = style_features
        x = self.conv1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.act2(x)
        x = self.conv3(x)
        x = x.squeeze(dim=-1) # (B, C, H, W, 1) -> (B, C, H, W)
        return x

class KFeatureExtractorUnit_FC(nn.Module):
    def __init__(self, K, singleton_shape):
        super(KFeatureExtractorUnit_FC, self).__init__()

        # self.K = K
        # self.singleton_shape = singleton_shape
        # self.singleton_size = int(torch.prod(torch.tensor(self.singleton_shape)))

        self.fc1 = nn.Linear(in_features=K, out_features=4*K)
        self.act1 = nn.ReLU()

        self.fc2 = nn.Linear(in_features=4*K, out_features=1)

    def forward(self, style_features: torch.Tensor):
        # expected input shape: (BATCH, K, C, H, W)

        dims = range(len(style_features.shape))
        permute_in = (dims[0], *dims[2:], 1) # (B, K, C, H, W) -> (B, C, H, W, K)

        x = style_features.permute(permute_in)
        x = self.fc1(x)
        x = self.act1(x)
        x = self.fc2(x)
        x = x.squeeze(dim=-1) # (B, C, H, W, 1) -> (B, C, H, W)

        return x

class KFeatureExtractor(nn.Module):
    def __init__(self, K):
        super(KFeatureExtractor, self).__init__()
        self.style_feature_extractor = KFeatureExtractorUnit_FC(K=K, singleton_shape=style_feature_extractor_input_singleton)
        self.content_feature_extractor1 = KFeatureExtractorUnit_FC(K=K, singleton_shape=content_feature_extractor_1_input_singleton)
        self.content_feature_extractor2 = KFeatureExtractorUnit_FC(K=K, singleton_shape=content_feature_extractor_2_input_singleton)
        self.content_feature_extractor3 = KFeatureExtractorUnit_FC(K=K, singleton_shape=content_feature_extractor_3_input_singleton)
        self.content_feature_extractor4 = KFeatureExtractorUnit_FC(K=K, singleton_shape=content_feature_extractor_4_input_singleton)
        self.content_feature_extractor5 = KFeatureExtractorUnit_FC(K=K, singleton_shape=content_feature_extractor_5_input_singleton)

    def forward(self, style_features: torch.Tensor, content_features: list[torch.Tensor]):
        style_features = self.style_feature_extractor(style_features)

        content_features = [
            self.content_feature_extractor1(content_features[0]),
            self.content_feature_extractor2(content_features[1]),
            self.content_feature_extractor3(content_features[2]),
            self.content_feature_extractor4(content_features[3]),
            self.content_feature_extractor5(content_features[4]),
        ]

        return style_features, content_features

def report_parameters_count(name, model):
    count = 0
    for param in model.parameters():
        count += torch.prod(torch.tensor(param.size()))
    print(f"{name} Parameters: {count}")

if __name__ == '__main__':
    test_batch_num = 16
    test_k = 5

    model = KFeatureExtractor(test_k)
    report_parameters_count("StyleFeatureExtractor", model.style_feature_extractor)
    report_parameters_count("ContentFeatureExtractor1", model.content_feature_extractor1)
    report_parameters_count("ContentFeatureExtractor2", model.content_feature_extractor2)
    report_parameters_count("ContentFeatureExtractor3", model.content_feature_extractor3)
    report_parameters_count("ContentFeatureExtractor4", model.content_feature_extractor4)
    report_parameters_count("ContentFeatureExtractor5", model.content_feature_extractor5)
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
