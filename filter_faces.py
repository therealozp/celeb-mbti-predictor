from pathlib import Path

import torch
import numpy as np
from PIL import Image
from ultralytics import YOLO
import pandas as pd

# -------- config --------
MODEL_WEIGHTS = "yolov12l-face.pt"  # path to face model weights
INPUT_ROOT = Path("images")  # your existing images/{MBTI}/...
OUTPUT_ROOT = Path("faces_yolo")  # where cropped faces go
CONF_THRESH = 0.25  # min confidence for a face
TARGET_SIZE = 224  # final face size (square)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_image(path: Path) -> Image.Image | None:
    try:
        return Image.open(path).convert("RGB")
    except Exception as e:
        print(f"[WARN] Cannot open {path}: {e}")
        return None


def main():
    print(f"[INFO] Loading model {MODEL_WEIGHTS} on {DEVICE}")
    model = YOLO(MODEL_WEIGHTS)
    model.to(DEVICE)

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    records = []
    img_id = 0

    # iterate MBTI folders
    for mbti_dir in sorted(p for p in INPUT_ROOT.iterdir() if p.is_dir()):
        mbti = mbti_dir.name.upper()
        print(f"\n[INFO] Processing MBTI={mbti}")

        # jpg/png/jpeg support
        image_paths = (
            list(mbti_dir.rglob("*.jpg"))
            + list(mbti_dir.rglob("*.jpeg"))
            + list(mbti_dir.rglob("*.png"))
        )

        for img_path in image_paths:
            rel_in = img_path.relative_to(INPUT_ROOT)
            pil_img = load_image(img_path)
            if pil_img is None:
                continue

            # convert to numpy (RGB)
            img_np = np.array(pil_img)

            # run YOLO-face; returns list[Results], we take first
            results = model.predict(
                img_np, verbose=False, conf=CONF_THRESH, device=DEVICE
            )
            if not results:
                print(f"[SKIP] No results for {rel_in}")
                continue

            res = results[0]
            boxes = res.boxes  # Boxes object: xyxy, conf, etc.
            if boxes is None or len(boxes) == 0:
                print(f"[SKIP] No faces above conf={CONF_THRESH} in {rel_in}")
                continue

            xyxy = boxes.xyxy.cpu().numpy()  # (N, 4)
            conf = boxes.conf.cpu().numpy()  # (N,)

            # choose largest box by area
            areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
            best_idx = int(areas.argmax())

            x1, y1, x2, y2 = xyxy[best_idx]
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            c = float(conf[best_idx])

            # basic sanity
            if x2 <= x1 or y2 <= y1:
                print(f"[SKIP] Degenerate box in {rel_in}")
                continue

            # crop + resize face
            face = pil_img.crop((x1, y1, x2, y2))
            face = face.resize((TARGET_SIZE, TARGET_SIZE), Image.BICUBIC)

            out_dir = OUTPUT_ROOT / mbti
            out_dir.mkdir(parents=True, exist_ok=True)
            out_name = img_path.stem + "_face.jpg"
            out_path = out_dir / out_name
            face.save(out_path)

            rel_out = out_path.relative_to(OUTPUT_ROOT)
            records.append(
                {
                    "id": img_id,
                    "mbti": mbti,
                    "input_path": str(rel_in).replace("\\", "/"),
                    "face_path": str(rel_out).replace("\\", "/"),
                    "conf": c,
                    "bbox_x1": x1,
                    "bbox_y1": y1,
                    "bbox_x2": x2,
                    "bbox_y2": y2,
                }
            )
            img_id += 1
            print(f"[OK] {rel_in} -> {rel_out} (conf={c:.3f})")

    if records:
        df = pd.DataFrame(records)
        df.to_csv("faces_yolo_metadata.csv", index=False)
        print(f"\n[DONE] Saved {len(records)} face crops.")
    else:
        print("\n[WARN] No faces saved.")


if __name__ == "__main__":
    main()
