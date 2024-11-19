import torch
import torch.nn as nn

class KFeatureExtractorUnit(nn.Module):
    def __init__(self, K, singleton_shape):
        super(KFeatureExtractorUnit, self).__init__()

        self.K = K
        self.singleton_shape = singleton_shape

        self.singleton_size = int(torch.prod(torch.tensor(self.singleton_shape)))
        self.fc = nn.Linear(self.K * self.singleton_size, self.singleton_size)

    def forward(self, style_features: torch.Tensor):
        # flatten the input tensor
        x = style_features.flatten()
        x: torch.Tensor = self.fc(x)  # pass through the fully connected layer
        x = x.reshape(self.singleton_shape)  # reshape the tensor to the original shape
        return x

class KFeatureExtractor(nn.Module):
    def __init__(self, K):
        super(KFeatureExtractor, self).__init__()
        self.style_feature_extractor = KFeatureExtractorUnit(K=K, singleton_shape=(3, 3, 1024))
        self.content_feature_extractor1 = KFeatureExtractorUnit(K=K, singleton_shape=(2, 3, 96, 96))
        self.content_feature_extractor2 = KFeatureExtractorUnit(K=K, singleton_shape=(2, 64, 48, 48))
        self.content_feature_extractor3 = KFeatureExtractorUnit(K=K, singleton_shape=(2, 128, 24, 24))
        self.content_feature_extractor4 = KFeatureExtractorUnit(K=K, singleton_shape=(2, 256, 12, 12))
        self.content_feature_extractor5 = KFeatureExtractorUnit(K=K, singleton_shape=(2, 256, 12, 12))

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