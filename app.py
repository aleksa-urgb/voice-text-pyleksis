from fastapi import FastAPI
from fastapi.responses import FileResponse
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
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")


# ======================
# Schemas
# ======================
class TTSReq(BaseModel):
    text: str


class SignReq(BaseModel):
    image_base64: str


# ======================
# Routes: UI + TTS
# ======================
@app.get("/")
def home():
    return FileResponse("static/index.html")


@app.post("/tts")
def tts(req: TTSReq):
    text = (req.text or "").strip()
    if not text:
        text = " "

    out_dir = "tmp_audio"
    os.makedirs(out_dir, exist_ok=True)

    filename = f"{uuid.uuid4().hex}.mp3"
    path = os.path.join(out_dir, filename)

    tts_obj = gTTS(text=text, lang="id")
    tts_obj.save(path)

    return FileResponse(path, media_type="audio/mpeg", filename="tts.mp3")


# ======================
# MediaPipe Tasks: Hand Landmarker
# ======================
MODEL_PATH = os.path.join("models", "hand_landmarker.task")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model tidak ditemukan: {MODEL_PATH}\n"
        "Taruh file hand_landmarker.task di folder models/ terlebih dahulu."
    )

BaseOptions = python.BaseOptions
HandLandmarker = vision.HandLandmarker
HandLandmarkerOptions = vision.HandLandmarkerOptions
VisionRunningMode = vision.RunningMode

# Create once (reused)
_hand_options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=1,
)
_hand_landmarker = HandLandmarker.create_from_options(_hand_options)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
AZ_MODEL_PATH = os.path.join(BASE_DIR, "models", "sign_az.pt")
AZ_LABELS_PATH = os.path.join(BASE_DIR, "models", "labels_az.json")

az_model = None
az_labels = None

if os.path.isfile(AZ_MODEL_PATH) and os.path.isfile(AZ_LABELS_PATH):
    az_model = torch.jit.load(AZ_MODEL_PATH, map_location="cpu")
    az_model.eval()
    with open(AZ_LABELS_PATH, "r", encoding="utf-8") as f:
        az_labels = json.load(f)

# ======================
# Helpers
# ======================
def decode_data_url_image(data_url: str) -> np.ndarray | None:
    """
    data_url: "data:image/jpeg;base64,...."
    returns BGR image (OpenCV)
    """
    try:
        _, encoded = data_url.split(",", 1)
        img_bytes = base64.b64decode(encoded)
        arr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return img
    except Exception:
        return None


def extract_hand_landmarks(img_bgr: np.ndarray):
    """
    Return list of 21 landmarks (each has x,y,z) or None.
    """
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    result = _hand_landmarker.detect(mp_image)
    if not result.hand_landmarks:
        return None

    # first detected hand
    return result.hand_landmarks[0]


def finger_states(lm) -> dict:
    """
    lm: list of 21 landmarks from MediaPipe Tasks (each has .x .y .z)
    Heuristic "finger up" states. Works best when palm faces camera.
    """
    def up(tip: int, pip: int) -> bool:
        return lm[tip].y < lm[pip].y  # smaller y = higher

    states = {
        "index": up(8, 6),
        "middle": up(12, 10),
        "ring": up(16, 14),
        "pinky": up(20, 18),
    }
    return states


def rule_based_letter(lm) -> tuple[str, float]:
    """
    Demo ONLY (bukan BISINDO final).
    Ini sekadar buktiin pipeline realtime jalan:
    kamera -> landmark -> label -> frontend -> TTS.

    Nanti kalau dataset sudah ada, fungsi ini diganti model.predict().
    """
    s = finger_states(lm)
    up_count = int(s["index"]) + int(s["middle"]) + int(s["ring"]) + int(s["pinky"])

    # A: semua turun (kepalan)
    if up_count == 0:
        return ("A", 0.70)

    # B / 5: semua naik (empat jari)
    if s["index"] and s["middle"] and s["ring"] and s["pinky"]:
        return ("B", 0.70)

    # V / 2: index + middle
    if s["index"] and s["middle"] and (not s["ring"]) and (not s["pinky"]):
        return ("V", 0.65)

    # 1: hanya index
    if s["index"] and (not s["middle"]) and (not s["ring"]) and (not s["pinky"]):
        return ("1", 0.55)

    return ("", 0.0)


# ======================
# Route: Sign predict
# ======================
@app.post("/sign/predict")
def sign_predict(req: SignReq):
    img = decode_data_url_image(req.image_base64)
    if img is None:
        return {"label": "", "confidence": 0.0}

    lm = extract_hand_landmarks(img)
    if lm is None:
        return {"label": "", "confidence": 0.0}

    if az_model is None or not az_labels:
        return {"label": "", "confidence": 0.0, "note": "Model A-Z belum ada. Jalankan extract_az.py lalu train_az.py"}

    feats = []
    for p in lm:
        feats.extend([p.x, p.y, p.z])
    x = torch.tensor(feats, dtype=torch.float32).unsqueeze(0)  # (1,63)

    with torch.no_grad():
        logits = az_model(x)
        probs = F.softmax(logits, dim=1)
        conf, idx = torch.max(probs, dim=1)

    label = az_labels[int(idx.item())]
    return {"label": label, "confidence": float(conf.item())}