import torch.nn as nn
from ultralytics import YOLO


class Yolov8nBackboneClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        yolo_model = YOLO("yolov8n.pt")
        self.backbone = yolo_model.model.model[:6]

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(128, 1), nn.Sigmoid()
        )

        # freeze all of the backbone layers
        for param in self.backbone.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = self.backbone(x)
        return self.classifier(x)
