import torch.nn as nn
from ultralytics import YOLO


class Yolov8nBackboneClassifierV2(nn.Module):
    def __init__(self):
        super().__init__()
        yolo_model = YOLO("yolov8n.pt")
        self.backbone = yolo_model.model.model[:9]

        # freeze all of the backbone layers
        for param in self.backbone.parameters():
            param.requires_grad = False

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.backbone(x)
        return self.classifier(x)
