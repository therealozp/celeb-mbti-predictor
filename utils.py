import pandas as pd
import numpy as np


def assign_splits(
    file_path: str = "faces_yolo_metadata.csv",
    id_column: str = "id",
    label_column: str = "mbti",
    seed: int = 42,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
):
    assert 0.0 < train_ratio < 1.0, "train_ratio must be in (0, 1)"
    assert 0.0 <= val_ratio < 1.0, "val_ratio must be in [0, 1)"
    assert train_ratio + val_ratio < 1.0, "train_ratio + val_ratio must be < 1.0"

    df = pd.read_csv(file_path)

    ID_COL = id_column
    LABEL_COL = label_column

    train_r, val_r = train_ratio, val_ratio
    rng = np.random.default_rng(seed)

    splits = {}

    for mbti, group in df.groupby(LABEL_COL):
        ids = group[ID_COL].unique()
        rng.shuffle(ids)

        n = len(ids)
        t = max(1, int(train_r * n))
        v = max(1, int(val_r * n)) if n >= 3 else 0
        splits_mbti = (
            {id_: "train" for id_ in ids[:t]}
            | {id_: "val" for id_ in ids[t : t + v]}
            | {id_: "test" for id_ in ids[t + v :]}
        )
        splits.update(splits_mbti)

    df["split"] = df[ID_COL].map(splits)
    df.to_csv(file_path, index=False)
    print(df["split"].value_counts())


def get_label_mapping(file_path, label_column: str = "mbti"):
    df = pd.read_csv(file_path)
    LABEL_COL = label_column

    mbti_to_idx = {mbti: idx for idx, mbti in enumerate(sorted(df[LABEL_COL].unique()))}
    return mbti_to_idx
