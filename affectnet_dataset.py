import torch
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
import os
from typing import Optional, Callable, Tuple


class AffectNetYoloDataset(Dataset):
    def __init__(
        self, root_dir: str, split: str = "train", transform: Optional[Callable] = None
    ):
        """
        Args:
            root_dir: Path to dataset root (containing train/ val/ test/ folders).
            split: One of 'train', 'val', or 'test'.
            transform: PyTorch transforms (must include Resize and ToTensor).
        """
        self.root = Path(root_dir) / split
        self.img_dir = self.root / "images"
        self.lbl_dir = self.root / "labels"
        self.transform = transform

        self.samples = []

        # 1. Scan for images and match with labels
        # We support common image extensions
        valid_exts = {".jpg", ".jpeg", ".png", ".bmp"}

        # List all files in images directory
        if not self.img_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {self.img_dir}")

        print(f"Scanning {self.img_dir}...")

        for img_file in os.listdir(self.img_dir):
            img_path = self.img_dir / img_file

            if img_path.suffix.lower() not in valid_exts:
                continue

            # Construct corresponding label path
            # img1.jpg -> img1.txt
            lbl_name = img_path.stem + ".txt"
            lbl_path = self.lbl_dir / lbl_name

            # Only add to dataset if both image and label exist
            if lbl_path.exists():
                # Store the paths to load lazily in __getitem__
                self.samples.append((str(img_path), str(lbl_path)))

        print(f"Found {len(self.samples)} valid image/label pairs in '{split}' split.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, lbl_path = self.samples[idx]

        # 1. Load Image
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a dummy tensor or handle gracefully (here we recurse to next)
            return self.__getitem__((idx + 1) % len(self))

        img_w, img_h = image.size

        # 2. Load Label & Bounding Box
        # Format: class_id x_center y_center width height (normalized 0-1)
        box_data = []
        with open(lbl_path, "r") as f:
            lines = f.readlines()
            if len(lines) > 0:
                # We take the first face found in the file
                # (AffectNet usually has 1 face per crop, but YOLO format supports many)
                box_data = lines[0].strip().split()

        if not box_data:
            # Handle empty label file -> Use entire image
            # This acts as a fallback to prevent crashing
            class_id = 0  # Default or 'Neutral'
            left, top, right, bottom = 0, 0, img_w, img_h
        else:
            class_id = int(box_data[0])

            # YOLO Normalized Float Values
            n_xc = float(box_data[1])
            n_yc = float(box_data[2])
            n_w = float(box_data[3])
            n_h = float(box_data[4])

            # 3. Convert YOLO (Norm Center) -> PIL (Abs Coordinates)
            # x_center_px = n_xc * img_w
            # width_px    = n_w * img_w
            # left        = x_center_px - (width_px / 2)

            x1 = (n_xc - n_w / 2) * img_w
            y1 = (n_yc - n_h / 2) * img_h
            x2 = (n_xc + n_w / 2) * img_w
            y2 = (n_yc + n_h / 2) * img_h

            # Clamp coordinates to stay within image bounds
            left = max(0, x1)
            top = max(0, y1)
            right = min(img_w, x2)
            bottom = min(img_h, y2)

            # Safety check: if box is degenerate (0 width), use full image
            if right <= left or bottom <= top:
                left, top, right, bottom = 0, 0, img_w, img_h

        # 4. Crop the Face
        face_crop = image.crop((left, top, right, bottom))

        # 5. Apply Transforms (Resize, ToTensor, Normalize)
        if self.transform:
            face_crop = self.transform(face_crop)

        return face_crop, class_id


def get_affectnet_transforms():
    import torchvision.transforms as T

    return T.Compose(
        [
            T.Resize((224, 224)),  # ResNet requires fixed size
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
