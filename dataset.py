from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, Optional, Tuple, List

import pandas as pd
from PIL import Image
import torch
from torch import Tensor
from torch.utils.data import Dataset


class CelebMBTIDataset(Dataset):
    def __init__(
        self,
        root_dir: str | Path,
        metadata_csv: str | Path,
        split: Optional[str] = None,
        transform: Optional[Callable[[Image.Image], Tensor]] = None,
        mbti_to_idx: Optional[Dict[str, int]] = None,
        label_column: str = "mbti",
        path_column: str = "face_path",
    ) -> None:
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.label_column = label_column
        self.path_column = path_column

        df = pd.read_csv(metadata_csv)

        if self.label_column not in df.columns:
            raise ValueError(
                f"Label column '{self.label_column}' not found in metadata."
            )
        if self.path_column not in df.columns:
            raise ValueError(f"Path column '{self.path_column}' not found in metadata.")

        # optional split filtering
        if split is not None:
            if "split" not in df.columns:
                raise ValueError(
                    "Requested a split, but l'split' column is not in metadata."
                )
            df = df[df["split"] == split].reset_index(drop=True)

        # normalize labels to uppercase (INTJ, ENFP, etc.)
        df[self.label_column] = df[self.label_column].astype(str).str.upper()

        self.mbti_to_idx = mbti_to_idx

        # build internal samples list
        samples: List[Tuple[Path, int]] = []
        for _, row in df.iterrows():
            label_str = row[self.label_column]
            if label_str not in self.mbti_to_idx:
                # skip unexpected label if mapping is fixed
                continue

            img_rel = Path(str(row[self.path_column]))
            img_path = self.root_dir / img_rel
            label_idx = self.mbti_to_idx[label_str]
            samples.append((img_path, label_idx))

        if not samples:
            raise RuntimeError("No valid samples found after filtering.")

        self.samples = samples
        self.num_classes = len(self.mbti_to_idx)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[Tensor, int]:
        img_path, label = self.samples[idx]

        # PIL load
        img = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        # img: Tensor(C, H, W), label: int
        return img, label


class CelebMBTIMultiTargetDataset(Dataset):
    def __init__(
        self,
        root_dir: str | Path,
        metadata_csv: str | Path,
        split: Optional[str] = None,
        transform: Optional[Callable[[Image.Image], Tensor]] = None,
        label_column: str = "mbti",
        path_column: str = "face_path",
    ) -> None:
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.label_column = label_column
        self.path_column = path_column

        df = pd.read_csv(metadata_csv)

        if self.label_column not in df.columns:
            raise ValueError(f"Label column '{self.label_column}' not found.")
        if self.path_column not in df.columns:
            raise ValueError(f"Path column '{self.path_column}' not found.")

        # Optional split filtering
        if split is not None:
            if "split" not in df.columns:
                raise ValueError(
                    "Requested a split, but 'split' column is not in metadata."
                )
            df = df[df["split"] == split].reset_index(drop=True)

        # Normalize labels to uppercase
        df[self.label_column] = df[self.label_column].astype(str).str.upper()

        # Build internal samples list
        # We store: (Path, FloatTensor[4])
        samples: List[Tuple[Path, Tensor]] = []

        for _, row in df.iterrows():
            label_str = row[self.label_column]

            # Sanity check: MBTI must be 4 chars (e.g., "INTJ")
            if len(label_str) != 4:
                continue

            img_rel = Path(str(row[self.path_column]))
            img_path = self.root_dir / img_rel

            # --- BINARY CONVERSION LOGIC ---
            # Mapping Convention:
            # 0: E=0 / I=1
            # 1: N=0 / S=1
            # 2: T=0 / F=1
            # 3: J=0 / P=1

            # We verify against the specific letter to be safe
            is_introvert = 1.0 if "I" in label_str else 0.0
            is_sensing = 1.0 if "S" in label_str else 0.0
            is_feeling = 1.0 if "F" in label_str else 0.0
            is_perceiver = 1.0 if "P" in label_str else 0.0

            # Create the vector [I, S, F, P]
            target_tensor = torch.tensor(
                [is_introvert, is_sensing, is_feeling, is_perceiver],
                dtype=torch.float32,
            )

            samples.append((img_path, target_tensor))

        if not samples:
            raise RuntimeError("No valid samples found after filtering.")

        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        img_path, label_vec = self.samples[idx]

        # PIL load
        try:
            img = Image.open(img_path).convert("RGB")
        except (OSError, FileNotFoundError):
            # Handle missing/corrupt images gracefully if needed
            # For now, we assume data integrity, but you might want to skip or log this
            raise RuntimeError(f"Failed to load image: {img_path}")

        if self.transform is not None:
            img = self.transform(img)

        # Returns:
        # img: Tensor(C, H, W)
        # label_vec: Tensor(4) -> [I, S, F, P]
        return img, label_vec
