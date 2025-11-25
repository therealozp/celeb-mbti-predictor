import cv2
import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision import models
from ultralytics import YOLO
from PIL import Image
import numpy as np

import time
from models import MBTISingleHeadMulticlass

# --- Configuration ---
YOLO_WEIGHTS = "yolov12s-face.pt"  # Face detection model
MBTI_WEIGHTS = "resnet_mbti_classification.pth"  # Path to your saved 16-class model
CONF_THRESH = 0.5  # YOLO confidence
TARGET_SIZE = 224  # ResNet input size
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Standard 16 MBTI Classes (Alphabetical order usually)
MBTI_CLASSES = [
    "ENFJ",
    "ENFP",
    "ENTJ",
    "ENTP",
    "ESFJ",
    "ESFP",
    "ESTJ",
    "ESTP",
    "INFJ",
    "INFP",
    "INTJ",
    "INTP",
    "ISFJ",
    "ISFP",
    "ISTJ",
    "ISTP",
]

# Standard ImageNet Transforms
val_transform = T.Compose(
    [
        T.Resize((TARGET_SIZE, TARGET_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def load_mbti_model(weights_path, device):
    """
    Loads the ResNet18 model architecture and weights for 16 classes.
    """
    print(f"[INFO] Loading MBTI Model from {weights_path}...")
    model = MBTISingleHeadMulticlass(num_classes=16)

    try:
        model.load_state_dict(torch.load(weights_path, map_location=device))
    except FileNotFoundError:
        print(
            f"[WARNING] Weights file '{weights_path}' not found. Using random weights for demo."
        )

    model.to(device)
    model.eval()
    return model


def predict_single_face(model, face_pil, device):
    """
    Takes a single PIL face image, transforms it, and returns the
    predicted class name and confidence score.
    """
    # 1. Transform: PIL -> Tensor [C, H, W] -> Batch [1, C, H, W]
    img_tensor = val_transform(face_pil).unsqueeze(0).to(device)

    # 2. Inference
    with torch.no_grad():
        logits = model(img_tensor)

        # 3. Softmax for probabilities
        probs = torch.softmax(logits, dim=1)

        # 4. Get Max Confidence and Index
        conf, pred_idx = torch.max(probs, dim=1)

        # Extract values
        conf_score = conf.item()
        pred_label = MBTI_CLASSES[pred_idx.item()]

    return pred_label, conf_score


def main():
    # 1. Load Models
    try:
        face_model = YOLO(YOLO_WEIGHTS)
        face_model.to(DEVICE)
    except Exception as e:
        print(f"[ERROR] Failed to load YOLO: {e}")
        return

    mbti_model = load_mbti_model(MBTI_WEIGHTS, DEVICE)

    # 2. Start Camera
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("[ERROR] Could not open webcam.")
        return

    print("[INFO] Camera started. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # YOLO Detection
        start = time.time()
        results = face_model(frame, verbose=False, conf=CONF_THRESH)
        print(f"[DEBUG] YOLO Inference Time: {time.time() - start:.3f} seconds")
        detections = results[0].boxes

        # --- PROCESS ONLY THE FIRST FACE ---
        if detections is not None and len(detections) > 0:
            # Take the first box (index 0)
            box = detections[0]
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Clip to frame
            h, w, _ = frame.shape
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            # Crop
            face_crop_bgr = frame[y1:y2, x1:x2]

            if face_crop_bgr.size > 0:
                # Prepare for Model (BGR -> RGB -> PIL)
                face_crop_rgb = cv2.cvtColor(face_crop_bgr, cv2.COLOR_BGR2RGB)
                pil_face = Image.fromarray(face_crop_rgb)

                # --- PREDICT ---
                start = time.time()
                pred_label, conf = predict_single_face(mbti_model, pil_face, DEVICE)
                print(
                    f"[DEBUG] MBTI Prediction Time: {time.time() - start:.3f} seconds"
                )

                # --- DRAW ---
                # Green box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Label with confidence
                label_text = f"{pred_label} ({conf*100:.1f}%)"
                cv2.putText(
                    frame,
                    label_text,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                )

        cv2.imshow("MBTI Predictor", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
