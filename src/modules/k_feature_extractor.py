import torch
import torch.nn as nn

style_feature_extractor_input_singleton = (3, 3, 1024)
content_feature_extractor_1_input_singleton = (3, 96, 96)
content_feature_extractor_2_input_singleton = (64, 48, 48)
content_feature_extractor_3_input_singleton = (128, 24, 24)
content_feature_extractor_4_input_singleton = (256, 12, 12)
content_feature_extractor_5_input_singleton = (256, 12, 12)

class KFeatureExtractorUnit(nn.Module):
    def __init__(self, K, singleton_shape):
        super(KFeatureExtractorUnit, self).__init__()

        self.K = K
        self.singleton_shape = singleton_shape
        self.singleton_size = int(torch.prod(torch.tensor(self.singleton_shape)))

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
        return x

class KFeatureExtractor(nn.Module):
    def __init__(self, K):
        super(KFeatureExtractor, self).__init__()
        self.style_feature_extractor = KFeatureExtractorUnit(K=K, singleton_shape=(3, 3, 1024))
        self.content_feature_extractor1 = KFeatureExtractorUnit(K=K, singleton_shape=(3, 96, 96))
        self.content_feature_extractor2 = KFeatureExtractorUnit(K=K, singleton_shape=(64, 48, 48))
        self.content_feature_extractor3 = KFeatureExtractorUnit(K=K, singleton_shape=(128, 24, 24))
        self.content_feature_extractor4 = KFeatureExtractorUnit(K=K, singleton_shape=(256, 12, 12))
        self.content_feature_extractor5 = KFeatureExtractorUnit(K=K, singleton_shape=(256, 12, 12))

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

def report_parameters_count(model):
    count = 0
    for param in model.parameters():
        count += torch.prod(torch.tensor(param.size()))
    print(f"KFeatureExtractor Parameters: {count}")

if __name__ == '__main__':
    model = KFeatureExtractor(5)
    report_parameters_count(model.style_feature_extractor)
    report_parameters_count(model.content_feature_extractor1)
    report_parameters_count(model.content_feature_extractor2)
    report_parameters_count(model.content_feature_extractor3)
    report_parameters_count(model.content_feature_extractor4)
    report_parameters_count(model.content_feature_extractor5)
    report_parameters_count(model)
