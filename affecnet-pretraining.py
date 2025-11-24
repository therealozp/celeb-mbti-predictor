import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pathlib import Path
from PIL import Image
import os
import time
from tqdm import tqdm
from typing import Optional, Callable, Tuple

from models import MBTIEfficientNetV2Small


# --- 1. The Dataset Class (YOLO Format) ---
class AffectNetYoloDataset(Dataset):
    def __init__(self, root_dir: str, split: str = "train", transform=None):
        self.root = Path(root_dir) / split
        self.img_dir = self.root / "images"
        self.lbl_dir = self.root / "labels"
        self.transform = transform
        self.samples = []

        valid_exts = {".jpg", ".jpeg", ".png", ".bmp"}

        # Scan directory
        if not self.img_dir.exists():
            print(f"Warning: {self.img_dir} not found. Skipping scan.")
            return

        print(f"Scanning {self.img_dir}...")
        for img_file in os.listdir(self.img_dir):
            img_path = self.img_dir / img_file
            if img_path.suffix.lower() not in valid_exts:
                continue

            lbl_name = img_path.stem + ".txt"
            lbl_path = self.lbl_dir / lbl_name

            if lbl_path.exists():
                self.samples.append((str(img_path), str(lbl_path)))

        print(f"Found {len(self.samples)} valid samples in '{split}'.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, lbl_path = self.samples[idx]

        try:
            image = Image.open(img_path).convert("RGB")
        except:
            return self.__getitem__((idx + 1) % len(self))
        # Parse YOLO Label
        box_data = []
        with open(lbl_path, "r") as f:
            lines = f.readlines()
            if len(lines) > 0:
                box_data = lines[0].strip().split()

        if not box_data:
            class_id = 0
        else:
            class_id = int(box_data[0])

        if self.transform:
            image = self.transform(image)

        return image, class_id


def train_affectnet(
    root_dir, epochs=15, batch_size=64, lr=1e-4, save_path="resnet18_affectnet_best.pth"
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # A. Transformations (ResNet requires 224x224)
    stats = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    train_tf = T.Compose(
        [
            T.Resize((224, 224)),
            T.RandomHorizontalFlip(),
            T.ColorJitter(0.2, 0.2, 0.2, 0.1),
            T.ToTensor(),
            T.Normalize(*stats),
        ]
    )
    val_tf = T.Compose([T.Resize((224, 224)), T.ToTensor(), T.Normalize(*stats)])

    # B. Load Data
    train_ds = AffectNetYoloDataset(root_dir, "train", transform=train_tf)
    val_ds = AffectNetYoloDataset(root_dir, "valid", transform=val_tf)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=4
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4)

    model = MBTIEfficientNetV2Small(num_classes=8, freeze_backbone=True)

    model = model.to(device)

    # D. Setup Training
    criterion = nn.CrossEntropyLoss()

    # First, train only the final layer
    optimizer_classifier = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr
    )
    print("Training only the final layer for 5 epochs...")
    for epoch in range(5):
        # --- TRAIN ---
        model.train()
        run_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for x, y in pbar:
            x, y = x.to(device), y.to(device)

            optimizer_classifier.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer_classifier.step()

            run_loss += loss.item() * x.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += x.size(0)

            pbar.set_postfix(loss=loss.item())

        epoch_loss = run_loss / total
        epoch_acc = correct / total

        # --- VALIDATION LOOP ---
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        # No gradients needed for validation
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)

                outputs = model(x)
                loss = criterion(outputs, y)

                val_loss += loss.item() * x.size(0)

                # Calculate accuracy just for monitoring
                _, preds = torch.max(outputs, 1)
                correct += (preds == y).sum().item()
                total += y.size(0)

        avg_val_loss = val_loss / len(val_loader.dataset)
        val_acc = correct / total

        # --- LOGGING ---
        print(
            f"Epoch {epoch+1} | "
            f"Val Loss: {avg_val_loss:.4f} | "
            f"Val Acc: {val_acc:.3f} | "
        )

    ## Unfreeze all layers and continue training
    for param in model.parameters():
        param.requires_grad = True

    optimizer = optim.Adam(model.parameters(), lr=lr * 0.1, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=3)

    # E. Training Loop
    print(f"Starting training for {epochs} epochs...")
    best_val_loss = float("inf")
    for epoch in range(epochs):
        # --- TRAIN ---
        model.train()
        run_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for x, y in pbar:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            run_loss += loss.item() * x.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += x.size(0)

            pbar.set_postfix(loss=loss.item())

        epoch_loss = run_loss / total
        epoch_acc = correct / total

        # --- VALIDATION LOOP ---
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        # No gradients needed for validation
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)

                outputs = model(x)
                loss = criterion(outputs, y)

                val_loss += loss.item() * x.size(0)

                # Calculate accuracy just for monitoring
                _, preds = torch.max(outputs, 1)
                correct += (preds == y).sum().item()
                total += y.size(0)

        avg_val_loss = val_loss / len(val_loader.dataset)
        val_acc = correct / total

        # --- SCHEDULER STEP ---
        # Critical: Pass the metric we want to monitor (avg_val_loss)
        scheduler.step(avg_val_loss)

        # Check current LR (for printing)
        current_lr = optimizer.param_groups[0]["lr"]

        # --- LOGGING ---
        print(
            f"Epoch {epoch+1} | "
            f"Val Loss: {avg_val_loss:.4f} | "
            f"Val Acc: {val_acc:.3f} | "
            f"LR: {current_lr:.2e}"
        )

        # Save Best Model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), save_path)
            print("  >>> Best model saved.")

        print(
            f"Ep {epoch+1} | Loss: {epoch_loss:.4f} | Tr Acc: {epoch_acc:.3f} | Val Acc: {val_acc:.3f}"
        )

    print("Done.")


if __name__ == "__main__":
    import sys

    DATA_ROOT = "./affectnet-yolo-format/YOLO_format"
    if not os.path.exists(DATA_ROOT):
        print(
            f"Dataset root '{DATA_ROOT}' not found. Please prepare the dataset first."
        )
        sys.exit(1)
    train_affectnet(
        root_dir=DATA_ROOT, epochs=50, save_path="efficientnetv2_affectnet_best.pth"
    )
