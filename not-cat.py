import argparse
from PIL import Image
from utils.loader import NotCatDataset
from models.efficientnet_backbone import EfficientnetBackboneClassifier
from models.yolov8n_backbone import Yolov8nBackboneClassifier
from models.yolov8n_backbone_v2 import Yolov8nBackboneClassifierV2
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import os
from tqdm.auto import tqdm
from torch.amp import autocast, GradScaler

NUM_IMAGES_OF_EACH_CLASS = 1000


def train_yolo_backbone_v2():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    # ensure model parameters are float32
    model = Yolov8nBackboneClassifierV2().to(device).float()
    criterion = nn.BCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

    image_transform = transforms.Compose(
        [
            transforms.Resize((160, 160)),
            transforms.ToTensor(),
            transforms.ConvertImageDtype(torch.float32),
        ]
    )

    dataset = NotCatDataset(
        "./photos/cat",
        "./photos/non-cat",
        max_samples=NUM_IMAGES_OF_EACH_CLASS,
        preload=True,
        transform=image_transform,
    )

    train_loader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=True,
        num_workers=os.cpu_count(),
        pin_memory=torch.cuda.is_available(),
        prefetch_factor=2,
        persistent_workers=True,
    )

    model.train()
    # scaler = GradScaler("cpu" if device.type == "cpu" else "cuda")
    scaler = GradScaler("cpu")  # for now, we'll use CPU
    epochs = 10

    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch_idx, (inputs, labels) in enumerate(pbar):
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with autocast(device_type=device.type, enabled=True):
                outputs = model(inputs).squeeze()
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            running_loss += loss.item()
            predictions = (outputs > 0.5).float()
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

            pbar.set_postfix(
                {
                    "loss": f"{running_loss/(batch_idx+1):.3f}",
                    "acc": f"{100.0*correct/total:.1f}%",
                }
            )

        print(
            f"Epoch {epoch+1}: Loss={running_loss/len(train_loader):.3f}, Acc={100.0*correct/total:.1f}%"
        )

    torch.save(model.state_dict(), "yolov8n_v2_backbone_classifier.pth")
    dummy_input = torch.randn(1, 3, 160, 160).to(device)
    torch.onnx.export(model, dummy_input, "yolov8n_v2_backbone_classifier.onnx")

    return model


def train_yolo_backbone():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    # ensure model parameters are float32
    model = Yolov8nBackboneClassifier().to(device).float()
    criterion = nn.BCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

    image_transform = transforms.Compose(
        [
            transforms.Resize((160, 160)),
            transforms.ToTensor(),
            transforms.ConvertImageDtype(torch.float32),
        ]
    )

    dataset = NotCatDataset(
        "./photos/cat",
        "./photos/non-cat",
        max_samples=NUM_IMAGES_OF_EACH_CLASS,
        preload=True,
        transform=image_transform,
    )

    train_loader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=True,
        num_workers=os.cpu_count(),
        pin_memory=torch.cuda.is_available(),
        prefetch_factor=2,
        persistent_workers=True,
    )

    model.train()
    # scaler = GradScaler("cpu" if device.type == "cpu" else "cuda")
    scaler = GradScaler("cpu")  # for now, we'll use CPU
    epochs = 10

    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch_idx, (inputs, labels) in enumerate(pbar):
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with autocast(device_type=device.type, enabled=True):
                outputs = model(inputs).squeeze()
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            running_loss += loss.item()
            predictions = (outputs > 0.5).float()
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

            pbar.set_postfix(
                {
                    "loss": f"{running_loss/(batch_idx+1):.3f}",
                    "acc": f"{100.0*correct/total:.1f}%",
                }
            )

        print(
            f"Epoch {epoch+1}: Loss={running_loss/len(train_loader):.3f}, Acc={100.0*correct/total:.1f}%"
        )

    torch.save(model.state_dict(), "yolov8n_backbone_classifier.pth")
    dummy_input = torch.randn(1, 3, 160, 160).to(device)
    torch.onnx.export(model, dummy_input, "yolov8n_backbone_classifier.onnx")

    return model


def train_eff_bone():
    print("Training mode activated...")

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "cpu"  # for now, we'll use CPU
    torch.backends.cudnn.benchmark = True

    model = EfficientnetBackboneClassifier()

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=0.001,
        weight_decay=0.01,
    )
    criterion = nn.BCELoss()

    image_transform = transforms.Compose(
        [
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    dataset = NotCatDataset(
        "./photos/cat",
        "./photos/non-cat",
        max_samples=NUM_IMAGES_OF_EACH_CLASS,
        preload=True,
        transform=image_transform,
    )

    train_loader = DataLoader(
        dataset,
        batch_size=256,
        shuffle=True,
        num_workers=os.cpu_count(),
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )

    print(f"\nTraining on device: {device}")
    model = model.to(device)
    epochs = 5

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for inputs, labels in pbar:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            predictions = (outputs > 0.5).float()
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

            pbar.set_postfix(
                {
                    "loss": f"{running_loss/total:.3f}",
                    "acc": f"{100.0*correct/total:.1f}%",
                }
            )

        print(
            f"Epoch {epoch+1}: Loss={running_loss/total:.3f}, Acc={100.0*correct/total:.1f}%"
        )

    # save the model
    torch.save(model.state_dict(), "eff_backbone_classifier.pth")
    dummy_input = torch.randn(1, 3, 128, 128).to(device)
    torch.onnx.export(model, dummy_input, "eff_backbone_classifier.onnx")


def run_eff():
    print("Running mode activated...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = EfficientnetBackboneClassifier()
    model.load_state_dict(torch.load("eff_backbone_classifier.pth"))

    image_transform = transforms.Compose(
        [
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    for root, dirs, files in os.walk("example_photos"):
        for file in files:
            filepath = os.path.join(root, file)

            image = image_transform(Image.open(filepath)).unsqueeze(0).to(device)
            model = model.to(device)

            with torch.no_grad():
                model.eval()
                output = model(image).item()

            print(f"{file[:10]}:\t{'cat' if output > 0.5 else 'not cat'}")


def run_yolo():
    print("Running mode activated...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Yolov8nBackboneClassifier()
    model.load_state_dict(torch.load("yolov8n_backbone_classifier.pth"))

    image_transform = transforms.Compose(
        [
            transforms.Resize((160, 160)),
            transforms.ToTensor(),
            transforms.ConvertImageDtype(torch.float32),
        ]
    )

    for root, dirs, files in os.walk("example_photos"):
        for file in files:
            filepath = os.path.join(root, file)

            image = image_transform(Image.open(filepath)).unsqueeze(0).to(device)
            model = model.to(device)

            with torch.no_grad():
                model.eval()
                output = model(image).item()

            print(f"{file[:10]}:\t{'cat' if output > 0.5 else 'not cat'}")


def run_yolo_v2():
    print("Running mode activated...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Yolov8nBackboneClassifierV2()
    model.load_state_dict(torch.load("yolov8n_v2_backbone_classifier.pth"))

    image_transform = transforms.Compose(
        [
            transforms.Resize((160, 160)),
            transforms.ToTensor(),
            transforms.ConvertImageDtype(torch.float32),
        ]
    )

    for root, dirs, files in os.walk("example_photos"):
        for file in files:
            filepath = os.path.join(root, file)

            image = image_transform(Image.open(filepath)).unsqueeze(0).to(device)
            model = model.to(device)

            with torch.no_grad():
                model.eval()
                output = model(image).item()

            print(f"{file[:10]}:\t{'cat' if output > 0.5 else 'not cat'}")


class ModelType:
    EFF = "eff"
    YOLO = "yolo"
    YOLO_V2 = "yolo_v2"


def main():
    parser = argparse.ArgumentParser(description="Not-cat CLI application")
    parser.add_argument(
        "mode", choices=["train", "run"], help="Select mode: train or run"
    )
    all_models = [v for k, v in vars(ModelType).items() if not k.startswith("__")]
    parser.add_argument(
        "--model",
        choices=[v for k, v in vars(ModelType).items() if not k.startswith("__")],
        help=f"Select model: {all_models}",
    )

    args = parser.parse_args()

    if args.mode == "train":
        if args.model == ModelType.EFF:
            train_eff_bone()
        elif args.model == ModelType.YOLO:
            train_yolo_backbone()
        elif args.model == ModelType.YOLO_V2:
            train_yolo_backbone_v2()
        else:
            print("Invalid model type")
    else:
        if args.model == ModelType.EFF:
            run_eff()
        elif args.model == ModelType.YOLO:
            run_yolo()
        elif args.model == ModelType.YOLO_V2:
            run_yolo_v2()


if __name__ == "__main__":
    main()
