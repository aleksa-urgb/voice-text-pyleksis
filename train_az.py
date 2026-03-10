"""
train_az.py — Training model MLP untuk klasifikasi isyarat BISINDO A-Z
======================================================================

Fix utama:
  1. Normalisasi WRIST-RELATIVE: fitur menjadi invariant terhadap posisi
     dan skala tangan di frame → akurasi jauh lebih baik
  2. Simpan mean/std untuk dipakai saat inferensi di main.py

Cara pakai:
  python extract_az.py   ← dulu
  python train_az.py     ← lalu ini
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from collections import Counter

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

NPZ_PATH = os.path.join(BASE_DIR, "data", "processed_az.npz")
MODELS_DIR = os.path.join(BASE_DIR, "models")
OUT_MODEL = os.path.join(MODELS_DIR, "sign_az.pt")
OUT_LABELS = os.path.join(MODELS_DIR, "labels_az.json")
OUT_MEAN = os.path.join(MODELS_DIR, "norm_mean.npy")
OUT_STD = os.path.join(MODELS_DIR, "norm_std.npy")

# ── hyperparameter ────────────────────────────────────────────────
EPOCHS = 50
BATCH_SIZE = 64
LR = 1e-3
VAL_SPLIT = 0.2
SEED = 42
# ─────────────────────────────────────────────────────────────────


# ── load data ─────────────────────────────────────────────────────
if not os.path.isfile(NPZ_PATH):
    print(f"[ERROR] {NPZ_PATH} tidak ditemukan.")
    print("Jalankan dulu: python extract_az.py")
    sys.exit(1)

data = np.load(NPZ_PATH, allow_pickle=True)
X_raw = data["X"].astype(np.float32)  # (N, 63)  — raw x,y,z per landmark
y = data["y"].astype(np.int64)  # (N,)
labels = [str(x) for x in data["labels"]]

print(f"[INFO] Data dimuat: {len(y)} sampel, {len(labels)} label kandidat")


# ================================================================
#  NORMALISASI WRIST-RELATIVE
#  ─────────────────────────────────────────────────────────────
#  Landmark 0 = pergelangan tangan (wrist).
#  Dengan mengurangkan wrist dan membagi dengan span kita membuat
#  fitur TIDAK peduli:
#    - di mana tangan di frame (translasi)
#    - seberapa dekat/jauh tangan ke kamera (skala)
#
#  Ini adalah teknik standar yang dipakai di paper-paper isyarat
#  tangan. Tanpa ini, model mudah bingung karena posisi tangan
#  orang berbeda-beda.
# ================================================================
def wrist_relative(X: np.ndarray) -> np.ndarray:
    """
    X: (N, 63) — setiap baris = 21 landmark x [x, y, z]
    Return: (N, 63) — koordinat relatif terhadap wrist & dinormalisasi skala
    """
    # Reshape jadi (N, 21, 3)
    pts = X.reshape(-1, 21, 3)

    # Kurangi wrist (landmark 0) agar translasi-invariant
    wrist = pts[:, 0:1, :]  # (N, 1, 3)
    pts = pts - wrist

    # Normalisasi skala: bagi dengan jarak max dari wrist
    # Hindari pembagian nol dengan + 1e-6
    scale = np.max(np.abs(pts), axis=(1, 2), keepdims=True) + 1e-6
    pts = pts / scale

    return pts.reshape(-1, 63)


X = wrist_relative(X_raw)
print("[INFO] Wrist-relative normalization applied.")


# ── hapus kelas yang tidak punya data ────────────────────────────
counts = Counter(y.tolist())
valid_idxs = sorted(k for k, v in counts.items() if v >= 2)
old2new = {old: new for new, old in enumerate(valid_idxs)}
labels_used = [labels[i] for i in valid_idxs]

mask = np.isin(y, valid_idxs)
X = X[mask]
y_remapped = np.array([old2new[yi] for yi in y[mask]], dtype=np.int64)

num_classes = len(labels_used)
print(f"[INFO] Kelas: {num_classes} → {labels_used}")

if num_classes < 2:
    print("[ERROR] Minimal 2 kelas diperlukan.")
    sys.exit(1)


# ── normalisasi statistik (Z-score) ──────────────────────────────
#  Dilakukan SETELAH wrist-relative agar tidak double-transform.
mean = X.mean(axis=0)
std = X.std(axis=0) + 1e-8
X_norm = (X - mean) / std

# Simpan untuk dipakai saat inferensi
os.makedirs(MODELS_DIR, exist_ok=True)
np.save(OUT_MEAN, mean)
np.save(OUT_STD, std)
print(f"[OK] Normalisasi disimpan: norm_mean.npy, norm_std.npy")


# ── train/val split ───────────────────────────────────────────────
try:
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_norm, y_remapped, test_size=VAL_SPLIT, random_state=SEED, stratify=y_remapped
    )
except ValueError:
    print("[WARN] Stratify gagal, pakai split biasa.")
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_norm, y_remapped, test_size=VAL_SPLIT, random_state=SEED
    )

train_ds = TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr))
val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
val_dl = DataLoader(val_ds, batch_size=256, shuffle=False)

print(f"[INFO] Train: {len(train_ds)}  Val: {len(val_ds)}")


# ── model ─────────────────────────────────────────────────────────
class MLP(nn.Module):
    def __init__(self, in_dim: int = 63, num_classes: int = 26):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Device: {device}")

model = MLP(63, num_classes).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
crit = nn.CrossEntropyLoss(label_smoothing=0.1)


# ── eval ──────────────────────────────────────────────────────────
def eval_acc():
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for xb, yb in val_dl:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb).argmax(dim=1)
            correct += (pred == yb).sum().item()
            total += yb.numel()
    return correct / max(total, 1)


# ── training loop ─────────────────────────────────────────────────
best_acc = 0.0
best_state = None

for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0.0
    for xb, yb in train_dl:
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad()
        loss = crit(model(xb), yb)
        loss.backward()
        opt.step()
        total_loss += loss.item()

    sched.step()
    acc = eval_acc()
    print(
        f"  Epoch {epoch:02d}/{EPOCHS} | loss={total_loss/len(train_dl):.4f} | val_acc={acc:.4f}"
    )

    if acc > best_acc:
        best_acc = acc
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

# ── simpan model ──────────────────────────────────────────────────
model.load_state_dict(best_state)
model.eval().to("cpu")

example = torch.zeros(1, 63)
ts = torch.jit.trace(model, example)
ts.save(OUT_MODEL)

with open(OUT_LABELS, "w", encoding="utf-8") as f:
    json.dump(labels_used, f, ensure_ascii=False, indent=2)

print()
print("=" * 50)
print(f"[OK] Model  : {OUT_MODEL}")
print(f"[OK] Labels : {OUT_LABELS}")
print(f"[OK] Best val_acc : {best_acc:.4f}  ({round(best_acc*100, 1)}%)")
print("=" * 50)
print("\nSekarang jalankan server:")
print("  uvicorn main:app --reload")
