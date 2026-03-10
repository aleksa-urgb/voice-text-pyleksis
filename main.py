"""
BISINDO Sign Language Recognition — FastAPI Backend
Fix: normalisasi wrist-relative + z-score diterapkan saat inferensi
"""

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from gtts import gTTS
import uuid
import os
import base64
import json
import torch
import torch.nn.functional as F
import cv2
import numpy as np
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ======================
# FastAPI setup
# ======================
app = FastAPI(title="BISINDO App")
app.mount("/static", StaticFiles(directory="static"), name="static")

os.makedirs("tmp_audio", exist_ok=True)
os.makedirs("models", exist_ok=True)


# ======================
# Schemas
# ======================
class TTSReq(BaseModel):
    text: str


class SignReq(BaseModel):
    image_base64: str


# ======================
# Routes: UI
# ======================
@app.get("/")
def home():
    return FileResponse("static/index.html")


# ======================
# Route: TTS
# ======================
@app.post("/tts")
def tts(req: TTSReq):
    text = (req.text or "").strip()
    if not text:
        text = " "
    filename = f"{uuid.uuid4().hex}.mp3"
    path = os.path.join("tmp_audio", filename)
    gTTS(text=text, lang="id").save(path)
    return FileResponse(path, media_type="audio/mpeg", filename="tts.mp3")


# ======================
# MediaPipe: Hand Landmarker (lazy init)
# ======================
MODEL_PATH = os.path.join("models", "hand_landmarker.task")
_hand_landmarker = None


def get_hand_landmarker():
    global _hand_landmarker
    if _hand_landmarker is not None:
        return _hand_landmarker
    if not os.path.exists(MODEL_PATH):
        return None
    BaseOptions = python.BaseOptions
    HandLandmarker = vision.HandLandmarker
    HandLandmarkerOptions = vision.HandLandmarkerOptions
    VisionRunningMode = vision.RunningMode
    opts = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=VisionRunningMode.IMAGE,
        num_hands=1,
    )
    _hand_landmarker = HandLandmarker.create_from_options(opts)
    return _hand_landmarker


# ======================
# ML Model A-Z
# ======================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
AZ_MODEL_PATH = os.path.join(BASE_DIR, "models", "sign_az.pt")
AZ_LABELS_PATH = os.path.join(BASE_DIR, "models", "labels_az.json")
NORM_MEAN_PATH = os.path.join(BASE_DIR, "models", "norm_mean.npy")
NORM_STD_PATH = os.path.join(BASE_DIR, "models", "norm_std.npy")

az_model = None
az_labels = None
norm_mean = None  # shape (63,)
norm_std = None  # shape (63,)


def load_az_model():
    global az_model, az_labels, norm_mean, norm_std
    if az_model is not None:
        return True

    if not (os.path.isfile(AZ_MODEL_PATH) and os.path.isfile(AZ_LABELS_PATH)):
        return False

    try:
        az_model = torch.jit.load(AZ_MODEL_PATH, map_location="cpu")
        az_model.eval()

        with open(AZ_LABELS_PATH, "r", encoding="utf-8") as f:
            az_labels = json.load(f)

        # ── Load normalisasi (WAJIB sama seperti saat training) ──
        if os.path.isfile(NORM_MEAN_PATH) and os.path.isfile(NORM_STD_PATH):
            norm_mean = np.load(NORM_MEAN_PATH).astype(np.float32)
            norm_std = np.load(NORM_STD_PATH).astype(np.float32)
            print(f"[OK] Normalisasi dimuat: mean shape {norm_mean.shape}")
        else:
            print("[WARN] norm_mean.npy / norm_std.npy tidak ditemukan.")
            print("       Jalankan ulang train_az.py untuk generate file normalisasi.")
            norm_mean = None
            norm_std = None

        print(f"[OK] Model loaded: {len(az_labels)} kelas → {az_labels}")
        return True

    except Exception as e:
        print(f"[ERROR] Load model gagal: {e}")
        return False


load_az_model()


# ======================
# Helpers
# ======================
def decode_data_url_image(data_url: str):
    try:
        encoded = data_url.split(",", 1)[1] if "," in data_url else data_url
        arr = np.frombuffer(base64.b64decode(encoded), np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception as e:
        print(f"[ERROR] decode image: {e}")
        return None


def extract_hand_landmarks(img_bgr):
    landmarker = get_hand_landmarker()
    if landmarker is None:
        return None
    try:
        rgb = np.ascontiguousarray(
            cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), dtype=np.uint8
        )
        result = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
        return result.hand_landmarks[0] if result.hand_landmarks else None
    except Exception as e:
        print(f"[ERROR] landmark: {e}")
        return None


def landmarks_to_features(lm) -> np.ndarray:
    """
    Convert 21 MediaPipe landmark → numpy array (63,) siap untuk model.

    Pipeline HARUS sama persis dengan train_az.py:
      1. Flatten raw x,y,z → (63,)
      2. Wrist-relative normalization  (posisi & skala invariant)
      3. Z-score normalization dengan mean/std yang disimpan saat training
    """
    # Step 1: flatten
    raw = np.array([[p.x, p.y, p.z] for p in lm], dtype=np.float32)  # (21, 3)

    # Step 2: wrist-relative
    wrist = raw[0]  # landmark 0 = wrist
    pts = raw - wrist  # kurangi wrist
    scale = np.max(np.abs(pts)) + 1e-6  # skala = max abs value
    pts = pts / scale  # normalisasi skala
    feats = pts.flatten()  # (63,)

    # Step 3: Z-score (pakai mean/std yang disimpan training)
    if norm_mean is not None and norm_std is not None:
        feats = (feats - norm_mean) / norm_std

    return feats


# ======================
# Route: Health check
# ======================
@app.get("/health")
def health():
    return {
        "hand_landmarker": os.path.exists(MODEL_PATH),
        "az_model": az_model is not None,
        "norm_loaded": norm_mean is not None,
        "labels_count": len(az_labels) if az_labels else 0,
    }


# ======================
# Route: Sign predict
# ======================
@app.post("/sign/predict")
def sign_predict(req: SignReq):
    # decode gambar
    img = decode_data_url_image(req.image_base64)
    if img is None:
        return JSONResponse(
            {"label": "", "confidence": 0.0, "error": "Gambar tidak valid"}
        )

    # cek hand landmarker
    if not os.path.exists(MODEL_PATH):
        return JSONResponse(
            {
                "label": "",
                "confidence": 0.0,
                "error": "hand_landmarker.task tidak ditemukan di models/",
            }
        )

    # deteksi tangan
    lm = extract_hand_landmarks(img)
    if lm is None:
        return JSONResponse(
            {"label": "", "confidence": 0.0, "error": "Tangan tidak terdeteksi"}
        )

    # coba load model kalau belum
    if az_model is None:
        load_az_model()

    if az_model is None or not az_labels:
        return JSONResponse(
            {
                "label": "",
                "confidence": 0.0,
                "error": "Model belum ada. Jalankan: python extract_az.py && python train_az.py",
            }
        )

    # ekstrak + normalisasi fitur (sama dengan pipeline training!)
    feats = landmarks_to_features(lm)
    x = torch.tensor(feats, dtype=torch.float32).unsqueeze(0)  # (1, 63)

    with torch.no_grad():
        logits = az_model(x)
        probs = F.softmax(logits, dim=1)
        conf, idx = torch.max(probs, dim=1)

    label = az_labels[int(idx.item())]
    return {"label": label, "confidence": float(conf.item())}
