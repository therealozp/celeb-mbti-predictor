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

        # fallback if your column is called 'four_letter'
        if self.label_column not in df.columns and "four_letter" in df.columns:
            self.label_column = "four_letter"

        if self.path_column not in df.columns:
            # fallback guess if you used another name
            for candidate in ["image_path", "filepath", "path"]:
                if candidate in df.columns:
                    self.path_column = candidate
                    break

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
                    "Requested a split, but 'split' column is not in metadata."
                )
            df = df[df["split"] == split].reset_index(drop=True)

        # normalize labels to uppercase (INTJ, ENFP, etc.)
        df[self.label_column] = df[self.label_column].astype(str).str.upper()

        # build or reuse MBTI→idx mapping
        if mbti_to_idx is None:
            unique_types: List[str] = sorted(df[self.label_column].unique().tolist())
            self.mbti_to_idx: Dict[str, int] = {
                t: i for i, t in enumerate(unique_types)
            }
        else:
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
