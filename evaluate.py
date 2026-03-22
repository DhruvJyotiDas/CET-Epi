import torch
from pathlib import Path

from src.models.cet_epi import CET_Epi
from src.data.chickenpox_loader import MultiScaleChickenpoxLoader

print("🚀 Starting evaluation...")

# ======================
# AUTO FIND LATEST CHECKPOINT
# ======================
base_path = Path("checkpoints/chickenpox")

all_runs = sorted(base_path.iterdir(), key=lambda x: x.stat().st_mtime)
latest_run = all_runs[-1]
checkpoint_path = latest_run / "best_model.pt"

print(f"📂 Using checkpoint: {checkpoint_path}")

# ======================
# LOAD DATA
# ======================
loader = MultiScaleChickenpoxLoader(lags=32)
train, test = loader.get_split(0.8)

sample = list(test)[0]

# 🔥 get normalization stats
mean = loader.mean
std = loader.std

print("Sample loaded:")
print("x shape:", sample.x.shape)
print("y shape:", sample.y.shape)

# ======================
# LOAD MODEL
# ======================
checkpoint = torch.load(checkpoint_path)

model = CET_Epi(
    n_micro=20,
    n_macro=5,
    in_channels=sample.x.shape[-1],
    hidden_dim=128
)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

print("✅ Model loaded")

# ======================
# RUN INFERENCE
# ======================
with torch.no_grad():
    pred, ei, extra = model(sample.x, sample.edge_index, return_all=True)

# ======================
# DENORMALIZE
# ======================
pred_denorm = pred.squeeze() * std + mean
y_denorm = sample.y * std + mean

print("\n📊 RESULTS (REAL SCALE)")
print("Prediction:", pred_denorm[:5])
print("Actual:", y_denorm[:5])

# ======================
# ERROR CHECK (REAL SCALE)
# ======================
error = (pred_denorm - y_denorm).abs().mean()
print("\n📉 Mean Absolute Error (REAL):", error.item())

# ======================
# EI CHECK
# ======================
print("\n🧠 EI Score:", (-ei).item())

# ======================
# CEO MATRIX CHECK
# ======================
print("\n🔬 Checking learned macro structure (S matrix)...")

S = extra["S"]

print("S shape:", S.shape)
print("\nFirst 5 rows of S:")
print(S[:5])

print("\nRow sums (should be ~1):")
print(S.sum(dim=1)[:5])

print("\nArgmax clusters (node → macro):")
print(torch.argmax(S, dim=1))

print("\n✅ Evaluation complete")
