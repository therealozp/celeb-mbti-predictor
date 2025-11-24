import torch

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
import seaborn as sns

from tqdm import tqdm


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


def plot_metric(train_values, val_values, title="Training Progress", ylabel="Metric"):
    """
    Plots a comparison between Training and Validation metrics over epochs.

    Args:
        train_values (list): List of training metric values (e.g., train_loss)
        val_values (list): List of validation metric values (e.g., val_loss)
        title (str): Title of the chart
        ylabel (str): Label for the Y-axis (e.g., "Loss" or "Accuracy")
    """
    epochs = range(1, len(train_values) + 1)

    plt.figure(figsize=(8, 5))

    # Plot Training (Blue solid line)
    plt.plot(epochs, train_values, "b-", linewidth=2, label=f"Training {ylabel}")

    # Plot Validation (Red dashed line)
    plt.plot(epochs, val_values, "r--", linewidth=2, label=f"Validation {ylabel}")

    plt.title(title, fontsize=14)
    plt.xlabel("Epochs", fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.legend(loc="best")
    plt.grid(True, alpha=0.3)  # Light grid for readability

    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(model, loader, device, class_names=None):
    """
    Computes and plots a confusion matrix for a PyTorch model.

    Args:
        model: The PyTorch model (already trained)
        loader: DataLoader (Validation or Test set)
        device: torch.device('cuda' or 'cpu')
        class_names: List of strings (e.g., ['Introvert', 'Extrovert'])
    """
    model.eval()
    all_preds = []
    all_labels = []

    print("Generating predictions for Confusion Matrix...")

    with torch.no_grad():
        for x, y in tqdm(loader, desc="Inference", leave=False):
            x, y = x.to(device), y.to(device)

            logits = model(x)

            # --- PREDICTION LOGIC ---
            # Check output dimension to determine prediction method
            if logits.shape[1] == 1:
                # Binary case with 1 output neuron (BCEWithLogitsLoss)
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).long().squeeze()
            else:
                # Multi-class or Binary with 2 neurons (CrossEntropyLoss)
                preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    # Compute Matrix
    cm = confusion_matrix(all_labels, all_preds)

    # Create Labels if not provided
    if class_names is None:
        class_names = [str(i) for i in range(cm.shape[0])]

    # Plotting
    plt.figure(figsize=(8, 6))

    # Heatmap
    # fmt='d' means integer formatting (no scientific notation)
    # cmap='Blues' is standard for confusion matrices
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )

    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.show()

    # Optional: Print normalized accuracy per class
    # Useful to see if one class is dominating (e.g. 99% acc on I, 0% on E)
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    print("\n--- Per-Class Accuracy ---")
    for i, name in enumerate(class_names):
        acc = cm_norm[i, i]
        print(f"{name}: {acc*100:.2f}%")
