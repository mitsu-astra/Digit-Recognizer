import argparse
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, classification_report
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt


# ----------------------------
# Reproducibility
# ----------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ----------------------------
# CNN Model
# ----------------------------
class MNISTCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # (1, 28, 28)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)   # -> (32, 28, 28)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # -> (64, 14, 14) after pool
        self.pool = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1) # -> (128, 7, 7) after pool
        self.dropout = nn.Dropout(0.25)

        self.fc1 = nn.Linear(128 * 7 * 7, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))      # (32, 28, 28)
        x = self.pool(F.relu(self.conv2(x)))  # (64, 14, 14)
        x = self.pool(F.relu(self.conv3(x)))  # (128, 7, 7)
        x = self.dropout(x)
        x = torch.flatten(x, 1)        # (batch, 128*7*7)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# ----------------------------
# Data
# ----------------------------
def get_dataloaders(batch_size=128, val_split=0.1, data_dir="./data"):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),  # Standard MNIST normalization
    ])

    train_full = datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
    testset = datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)

    val_size = int(len(train_full) * val_split)
    train_size = len(train_full) - val_size
    trainset, valset = random_split(train_full, [train_size, val_size])

    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    return train_loader, val_loader, test_loader


# ----------------------------
# Training & Evaluation
# ----------------------------
def train_one_epoch(model, loader, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in tqdm(loader, desc="Train", leave=False):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = F.cross_entropy(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(targets).sum().item()
        total += targets.size(0)

    return running_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, device, phase="Val"):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds, all_targets = [], []

    for inputs, targets in tqdm(loader, desc=phase, leave=False):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = F.cross_entropy(outputs, targets)

        running_loss += loss.item() * inputs.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(targets).sum().item()
        total += targets.size(0)

        all_preds.append(preds.cpu().numpy())
        all_targets.append(targets.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    return running_loss / total, correct / total, all_preds, all_targets


def plot_confusion_matrix(y_true, y_pred, save_path=None):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(cm, interpolation='nearest')
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(10),
        yticks=np.arange(10),
        xticklabels=list(range(10)),
        yticklabels=list(range(10)),
        ylabel='True label',
        title='Confusion Matrix'
    )
    ax.set_xlabel('Predicted label')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    return fig


# ----------------------------
# Predict single image
# ----------------------------
def preprocess_image_for_mnist(path):
    """
    Accepts: grayscale or RGB. Resizes to 28x28, converts to grayscale,
    inverts if background is black & digit is white? (MNIST is white digit on black),
    then normalizes the same way as training.
    """
    img = Image.open(path).convert("L")  # grayscale
    img = img.resize((28, 28))
    arr = np.array(img).astype(np.float32)
    # Heuristic: if background appears white, invert to match MNIST (white digit on black)
    if arr.mean() > 127:
        arr = 255 - arr
    arr = arr / 255.0
    arr = (arr - 0.1307) / 0.3081
    tensor = torch.tensor(arr).unsqueeze(0).unsqueeze(0)  # (1,1,28,28)
    return tensor


@torch.no_grad()
def predict(model_path, image_path, device="cpu"):
    model = MNISTCNN()
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state["model_state"])
    model.to(device)
    model.eval()

    x = preprocess_image_for_mnist(image_path).to(device)
    logits = model(x)
    prob = F.softmax(logits, dim=1).cpu().numpy()[0]
    pred = int(np.argmax(prob))
    return pred, prob


# ----------------------------
# Main
# ----------------------------
def main():
    parser = argparse.ArgumentParser(description="MNIST CNN Trainer/Tester")
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_dir", type=str, default="./checkpoints")
    parser.add_argument("--train", action="store_true", help="Run training")
    parser.add_argument("--test", action="store_true", help="Evaluate on test set")
    parser.add_argument("--predict_image", type=str, default="", help="Path to an image to predict")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device(args.device)

    model = MNISTCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    ckpt_path = Path(args.save_dir) / "mnist_cnn_best.pt"

    if args.train:
        train_loader, val_loader, test_loader = get_dataloaders(
            batch_size=args.batch_size, data_dir=args.data_dir
        )

        best_val_acc = 0.0
        patience = 3
        patience_counter = 0

        for epoch in range(1, args.epochs + 1):
            print(f"\nEpoch {epoch}/{args.epochs}")
            tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, device)
            val_loss, val_acc, _, _ = evaluate(model, val_loader, device, phase="Val")

            print(
                f"Train Loss: {tr_loss:.4f} | Train Acc: {tr_acc*100:.2f}% | "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}%"
            )

            # Early stopping + checkpoint
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                torch.save(
                    {"model_state": model.state_dict(), "val_acc": best_val_acc},
                    ckpt_path,
                )
                print(f"Saved new best model to {ckpt_path}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print("Early stopping triggered.")
                    break

        # Load best for final test eval
        if ckpt_path.exists():
            state = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(state["model_state"])

        test_loss, test_acc, y_pred, y_true = evaluate(model, test_loader, device, phase="Test")
        print(f"\nTest Loss: {test_loss:.4f} | Test Acc: {test_acc*100:.2f}%")

        # Reports
        print("\nClassification report:")
        print(classification_report(y_true, y_pred, digits=4))

        # Confusion Matrix
        fig = plot_confusion_matrix(y_true, y_pred, save_path=str(Path(args.save_dir) / "confusion_matrix.png"))
        plt.show()

    if args.test and not args.train:
        # If user just wants to test existing checkpoint
        assert ckpt_path.exists(), f"Checkpoint not found at {ckpt_path}"
        _, _, test_loader = get_dataloaders(data_dir=args.data_dir)
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state["model_state"])
        test_loss, test_acc, y_pred, y_true = evaluate(model, test_loader, device, phase="Test")
        print(f"\nTest Loss: {test_loss:.4f} | Test Acc: {test_acc*100:.2f}%")
        print("\nClassification report:")
        print(classification_report(y_true, y_pred, digits=4))
        fig = plot_confusion_matrix(y_true, y_pred)
        plt.show()

    if args.predict_image:
        assert ckpt_path.exists(), f"Checkpoint not found at {ckpt_path}"
        pred, prob = predict(str(ckpt_path), args.predict_image, device=device)
        print(f"Predicted: {pred}")
        print("Class probabilities:", np.round(prob, 4))


if __name__ == "__main__":
    main()
    