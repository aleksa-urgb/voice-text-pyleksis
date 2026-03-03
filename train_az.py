import os, json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
NPZ_PATH = os.path.join(BASE_DIR, "data", "processed_az.npz")

MODELS_DIR = os.path.join(BASE_DIR, "models")
OUT_MODEL = os.path.join(MODELS_DIR, "sign_az.pt")
OUT_LABELS = os.path.join(MODELS_DIR, "labels_az.json")

data = np.load(NPZ_PATH, allow_pickle=True)
X = data["X"].astype(np.float32)  # (N,63)
y = data["y"].astype(np.int64)    # (N,)
labels = [x for x in data["labels"]]

num_classes = len(labels)

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
val_ds   = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))

train_dl = DataLoader(train_ds, batch_size=64, shuffle=True)
val_dl   = DataLoader(val_ds, batch_size=128, shuffle=False)

class MLP(nn.Module):
    def __init__(self, in_dim=63, num_classes=26):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.net(x)

device = "cpu"
model = MLP(63, num_classes).to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
crit = nn.CrossEntropyLoss()

def eval_acc():
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for xb, yb in val_dl:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            pred = logits.argmax(dim=1)
            correct += (pred == yb).sum().item()
            total += yb.numel()
    return correct / max(total, 1)

best = 0.0
for epoch in range(1, 21):
    model.train()
    for xb, yb in train_dl:
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad()
        logits = model(xb)
        loss = crit(logits, yb)
        loss.backward()
        opt.step()

    acc = eval_acc()
    print(f"Epoch {epoch:02d} | val_acc={acc:.4f}")
    if acc > best:
        best = acc
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

model.load_state_dict(best_state)
os.makedirs(MODELS_DIR, exist_ok=True)

# Export TorchScript (biar gampang dipakai FastAPI)
example = torch.zeros(1, 63)
ts = torch.jit.trace(model, example)
ts.save(OUT_MODEL)

with open(OUT_LABELS, "w", encoding="utf-8") as f:
    json.dump(labels, f, ensure_ascii=False)

print("Saved model:", OUT_MODEL)
print("Saved labels:", OUT_LABELS)
print("Best val_acc:", best)