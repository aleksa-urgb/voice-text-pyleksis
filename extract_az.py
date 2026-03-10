"""
extract_az.py — Extract MediaPipe hand landmarks dari dataset BISINDO A-Z
========================================================================

Struktur folder Kaggle yang didukung (auto-detect):

  Opsi 1 (paling umum):
    data/raw/alfabet_bisindo/
      A/  ← langsung berisi gambar
        001.jpg
        002.jpg
      B/
        ...

  Opsi 2 (ada subfolder train/):
    data/raw/alfabet_bisindo/
      train/
        A/
        B/

  Opsi 3 (huruf huruf langsung di root):
    data/raw/alfabet_bisindo/
      a/   ← lowercase
      b/

Cara pakai:
  1. Download dataset BISINDO dari Kaggle
  2. Unzip ke  data/raw/alfabet_bisindo/
  3. Jalankan:  python extract_az.py
"""

import os
import json
import sys
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── konfigurasi path ──────────────────────────────────────────────
RAW_ROOT = os.path.join(BASE_DIR, "data", "raw", "alfabet_bisindo")
OUT_NPZ = os.path.join(BASE_DIR, "data", "processed_az.npz")
MODEL_PATH = os.path.join(BASE_DIR, "models", "hand_landmarker.task")
# ─────────────────────────────────────────────────────────────────


# ── cek model tersedia ────────────────────────────────────────────
if not os.path.exists(MODEL_PATH):
    print("=" * 60)
    print("[ERROR] hand_landmarker.task tidak ditemukan!")
    print(f"  Path yang dicek: {MODEL_PATH}")
    print()
    print("Download model di:")
    print(
        "  https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
    )
    print("Simpan ke:  models/hand_landmarker.task")
    print("=" * 60)
    sys.exit(1)

if not os.path.isdir(RAW_ROOT):
    print("=" * 60)
    print("[ERROR] Folder dataset tidak ditemukan!")
    print(f"  Path yang dicek: {RAW_ROOT}")
    print()
    print("Buat folder dan isi dengan dataset Kaggle:")
    print(f"  mkdir -p {RAW_ROOT}")
    print("  Lalu unzip dataset Kaggle ke sana.")
    print("=" * 60)
    sys.exit(1)


# ── auto-detect subfolder dataset ────────────────────────────────
def find_class_root(root: str) -> str:
    """
    Cari folder yang langsung berisi subfolder A-Z (atau a-z).
    Mendukung satu level kedalaman ekstra (train/, test/, dll).
    """
    labels_upper = {chr(ord("A") + i) for i in range(26)}
    labels_lower = {c.lower() for c in labels_upper}

    def has_alpha_dirs(path):
        try:
            children = {
                d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))
            }
            return bool(children & (labels_upper | labels_lower))
        except Exception:
            return False

    if has_alpha_dirs(root):
        return root

    # coba satu level dalam
    for sub in sorted(os.listdir(root)):
        candidate = os.path.join(root, sub)
        if os.path.isdir(candidate) and has_alpha_dirs(candidate):
            print(f"[INFO] Dataset ditemukan di subfolder: {sub}/")
            return candidate

    # fallback — kembalikan root
    print("[WARN] Struktur folder tidak dikenali, pakai root langsung.")
    return root


CLASS_ROOT = find_class_root(RAW_ROOT)
LABELS = [chr(ord("A") + i) for i in range(26)]

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ── setup MediaPipe ───────────────────────────────────────────────
BaseOptions = python.BaseOptions
HandLandmarker = vision.HandLandmarker
HandLandmarkerOptions = vision.HandLandmarkerOptions
VisionRunningMode = vision.RunningMode

opts = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=1,
)
detector = HandLandmarker.create_from_options(opts)
print("[OK] MediaPipe Hand Landmarker siap.")


def extract_landmarks(img_bgr) -> np.ndarray | None:
    """Return array shape (63,) atau None kalau tangan tidak terdeteksi."""
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    try:
        res = detector.detect(mp_image)
    except Exception as e:
        print(f"  [WARN] MediaPipe error: {e}")
        return None

    if not res.hand_landmarks:
        return None

    lm = res.hand_landmarks[0]
    feats = []
    for p in lm:
        feats.extend([p.x, p.y, p.z])
    return np.array(feats, dtype=np.float32)  # (63,)


# ── proses dataset ────────────────────────────────────────────────
X, y_list = [], []
miss = 0
total_imgs = 0

for idx, lab in enumerate(LABELS):
    # coba uppercase dulu, lalu lowercase
    class_dir = os.path.join(CLASS_ROOT, lab)
    if not os.path.isdir(class_dir):
        class_dir = os.path.join(CLASS_ROOT, lab.lower())
    if not os.path.isdir(class_dir):
        print(f"  [SKIP] Folder '{lab}' tidak ditemukan")
        continue

    imgs_in_class = [
        f for f in os.listdir(class_dir) if os.path.splitext(f)[1].lower() in IMG_EXTS
    ]

    if not imgs_in_class:
        print(f"  [SKIP] Folder '{lab}' kosong (tidak ada gambar)")
        continue

    ok = 0
    for fn in imgs_in_class:
        path = os.path.join(class_dir, fn)
        img = cv2.imread(path)
        if img is None:
            continue
        total_imgs += 1
        feats = extract_landmarks(img)
        if feats is None:
            miss += 1
            continue
        X.append(feats)
        y_list.append(idx)
        ok += 1

    print(f"  [{lab}] {ok}/{len(imgs_in_class)} berhasil diekstrak")

# ── simpan ────────────────────────────────────────────────────────
if not X:
    print("\n[ERROR] Tidak ada data yang berhasil diekstrak!")
    print("Pastikan gambar ada di folder dan tangan terdeteksi.")
    sys.exit(1)

X_arr = np.stack(X).astype(np.float32)
y_arr = np.array(y_list, dtype=np.int64)

os.makedirs(os.path.dirname(OUT_NPZ), exist_ok=True)
np.savez_compressed(OUT_NPZ, X=X_arr, y=y_arr, labels=np.array(LABELS))

print()
print("=" * 50)
print(f"[OK] Saved: {OUT_NPZ}")
print(f"  Total gambar dibaca : {total_imgs}")
print(f"  Berhasil diekstrak  : {len(y_list)}")
print(f"  Gagal (no hand)     : {miss}")
print(f"  Kelas ada datanya   : {len(set(y_list))}")
print("=" * 50)
print("\nLangkah berikutnya: python train_az.py")
