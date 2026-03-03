import os, json
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

RAW_DIR = os.path.join(BASE_DIR, "data", "raw", "alfabet_bisindo")
OUT_NPZ = os.path.join(BASE_DIR, "data", "processed_az.npz")
MODEL_PATH = os.path.join(BASE_DIR, "models", "hand_landmarker.task")

labels = [chr(ord("A")+i) for i in range(26)]

BaseOptions = python.BaseOptions
HandLandmarker = vision.HandLandmarker
HandLandmarkerOptions = vision.HandLandmarkerOptions
VisionRunningMode = vision.RunningMode

opts = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=1
)
detector = HandLandmarker.create_from_options(opts)

def extract_landmarks(img_bgr):
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    res = detector.detect(mp_image)
    if not res.hand_landmarks:
        return None
    lm = res.hand_landmarks[0]
    feats = []
    for p in lm:
        feats.extend([p.x, p.y, p.z])
    return np.array(feats, dtype=np.float32)  # (63,)

X, y = [], []
miss = 0

for idx, lab in enumerate(labels):
    class_dir = os.path.join(RAW_DIR, lab)
    if not os.path.isdir(class_dir):
        print("Skip (folder not found):", class_dir)
        continue

    for fn in os.listdir(class_dir):
        if not fn.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        path = os.path.join(class_dir, fn)
        img = cv2.imread(path)
        if img is None:
            continue
        feats = extract_landmarks(img)
        if feats is None:
            miss += 1
            continue
        X.append(feats)
        y.append(idx)

X = np.stack(X) if X else np.zeros((0,63), dtype=np.float32)
y = np.array(y, dtype=np.int64)

os.makedirs(os.path.dirname(OUT_NPZ), exist_ok=True)
np.savez_compressed(OUT_NPZ, X=X, y=y, labels=np.array(labels))

print("Saved:", OUT_NPZ)
print("Samples:", len(y), "Miss(no hand):", miss)