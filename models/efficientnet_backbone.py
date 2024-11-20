import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


class EfficientnetBackboneClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        self.features = nn.Sequential(*list(backbone.children())[:-1])
        self.classifier = nn.Sequential(
            nn.Dropout(0.2), nn.Linear(1280, 1), nn.Sigmoid()
        )

        # freeze most layers for faster training
        for param in self.features.parameters():
            param.requires_grad = False

        # only train the last few layers
        for param in list(self.features.parameters())[-4:]:
            param.requires_grad = True

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)
