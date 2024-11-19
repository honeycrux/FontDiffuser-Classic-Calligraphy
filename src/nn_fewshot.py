import torch
import torch.nn as nn

class StyleFeatureExtractor(nn.Module):
    def __init__(self):
        super(StyleFeatureExtractor, self).__init__()

        self.K = 5
        self.singleton_shape = (3, 3, 1024)

        self.singleton_size = int(torch.prod(torch.tensor(self.singleton_shape)).item())
        # fully connected layer: 5 * 3 * 3 * 1024 -> 3 * 3 * 1024
        self.fc = nn.Linear(self.K * self.singleton_size, self.singleton_size)

    def forward(self, style_features: torch.Tensor):
        # flatten the input tensor
        x = style_features.flatten()
        x: torch.Tensor = self.fc(x)  # pass through the fully connected layer
        x = x.reshape(self.singleton_shape)  # reshape the tensor to the original shape
        return x

class ContentFeatureExtractor1(nn.Module):
    def __init__(self):
        super(ContentFeatureExtractor1, self).__init__()
        self.K = 5
        self.singleton_shape = (2, 3, 96, 96)
        self.singleton_size = int(torch.prod(torch.tensor(self.singleton_shape)).item())
        self.fc = nn.Linear(self.K * self.singleton_size, self.singleton_size)

    def forward(self, content_features: torch.Tensor):
        x = content_features.flatten()
        x: torch.Tensor = self.fc(x)
        x = x.reshape(self.singleton_shape)
        return x

class ContentFeatureExtractor2(nn.Module):
    def __init__(self):
        super(ContentFeatureExtractor2, self).__init__()
        self.K = 5
        self.singleton_shape = (2, 64, 48, 48)
        self.singleton_size = int(torch.prod(torch.tensor(self.singleton_shape)).item())
        self.fc = nn.Linear(self.K * self.singleton_size, self.singleton_size)

    def forward(self, content_features: torch.Tensor):
        x = content_features.flatten()
        x: torch.Tensor = self.fc(x)
        x = x.reshape(self.singleton_shape)
        return x

class ContentFeatureExtractor3(nn.Module):
    def __init__(self):
        super(ContentFeatureExtractor3, self).__init__()
        self.K = 5
        self.singleton_shape = (2, 128, 24, 24)
        self.singleton_size = int(torch.prod(torch.tensor(self.singleton_shape)).item())
        self.fc = nn.Linear(self.K * self.singleton_size, self.singleton_size)

    def forward(self, content_features: torch.Tensor):
        x = content_features.flatten()
        x: torch.Tensor = self.fc(x)
        x = x.reshape(self.singleton_shape)
        return x

class ContentFeatureExtractor4(nn.Module):
    def __init__(self):
        super(ContentFeatureExtractor4, self).__init__()
        self.K = 5
        self.singleton_shape = (2, 256, 12, 12)
        self.singleton_size = int(torch.prod(torch.tensor(self.singleton_shape)).item())
        self.fc = nn.Linear(self.K * self.singleton_size, self.singleton_size)

    def forward(self, content_features: torch.Tensor):
        x = content_features.flatten()
        x: torch.Tensor = self.fc(x)
        x = x.reshape(self.singleton_shape)
        return x

ContentFeatureExtractor5 = ContentFeatureExtractor4