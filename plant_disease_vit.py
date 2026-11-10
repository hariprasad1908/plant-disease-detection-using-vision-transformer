"""
Plant Disease Detection using Vision Transformer (ViT)
=======================================================
Model     : Vision Transformer (ViT) – built from scratch in PyTorch
            *** ZERO convolutional layers – pure multi-head self-attention ***
Framework : PyTorch 2.x
"""

import os, sys, math, time, warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 65, flush=True)
print("  PLANT DISEASE DETECTION  –  Vision Transformer (ViT)", flush=True)
print("=" * 65, flush=True)
print(f"  PyTorch  : {torch.__version__}", flush=True)
print(f"  Device   : {DEVICE}", flush=True)
print(f"  NumPy    : {np.__version__}", flush=True)
print("=" * 65, flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# 1. Config
# ─────────────────────────────────────────────────────────────────────────────
IMG_SIZE         = 96        # 96x96 leaf images
PATCH_SIZE       = 12        # 12x12 patches -> 8x8 = 64 patches
NUM_PATCHES      = (IMG_SIZE // PATCH_SIZE) ** 2    # 64
PATCH_DIM        = PATCH_SIZE * PATCH_SIZE * 3      # 432
EMBED_DIM        = 128       # ViT embedding dimension
NUM_HEADS        = 4         # Attention heads (128 / 4 = 32 per head)
NUM_LAYERS       = 4         # 4 Transformer Encoder blocks
MLP_RATIO        = 3         # Hidden MLP dimension = 128 * 3 = 384
DROPOUT          = 0.1
BATCH_SIZE       = 32
EPOCHS           = 12
LR               = 5e-4
WEIGHT_DECAY     = 1e-4

CLASS_NAMES = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___healthy",
    "Corn___Common_rust",
    "Corn___healthy",
    "Grape___Black_rot",
    "Grape___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___healthy",
    "Tomato___Leaf_Mold",
    "Tomato___Yellow_Leaf_Curl_Virus",
]
NUM_CLASSES = len(CLASS_NAMES)

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULTS, exist_ok=True)

print(f"\n  Classes    : {NUM_CLASSES}", flush=True)
print(f"  Img size   : {IMG_SIZE}×{IMG_SIZE}", flush=True)
print(f"  Patch size : {PATCH_SIZE}×{PATCH_SIZE} ({NUM_PATCHES} patches/image)", flush=True)
print(f"  Embed dim  : {EMBED_DIM} | Heads: {NUM_HEADS} | Layers: {NUM_LAYERS}", flush=True)
print(f"  Epochs     : {EPOCHS} | Batch size: {BATCH_SIZE} | LR: {LR}", flush=True)
print("", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# 2. Fast synthetic dataset generation
# ─────────────────────────────────────────────────────────────────────────────
print("[1/5] Generating plant leaf disease dataset …", flush=True)

def generate_dataset(samples_per_class=80, size=96):
    """Vectorized and fast leaf generator with distinct disease signatures."""
    X_all = []
    y_all = []
    
    Y, X = np.ogrid[:size, :size]
    cx, cy = size // 2, size // 2

    for ci, name in enumerate(CLASS_NAMES):
        rng = np.random.default_rng(ci * 777 + 42)
        is_healthy = "healthy" in name
        
        # Color baseline per class
        if is_healthy:
            base_r, base_g, base_b = 0.10, 0.65, 0.10
        elif "rust" in name.lower():
            base_r, base_g, base_b = 0.60, 0.35, 0.05
        elif "rot" in name.lower():
            base_r, base_g, base_b = 0.25, 0.20, 0.15
        elif "scab" in name.lower() or "mold" in name.lower():
            base_r, base_g, base_b = 0.38, 0.40, 0.15
        elif "blight" in name.lower():
            base_r, base_g, base_b = 0.45, 0.30, 0.10
        elif "virus" in name.lower():
            base_r, base_g, base_b = 0.55, 0.58, 0.08
        else: # spot
            base_r, base_g, base_b = 0.35, 0.35, 0.12

        imgs = np.zeros((samples_per_class, size, size, 3), dtype=np.float32)
        
        for i in range(samples_per_class):
            # Elliptical leaf body with random rotation
            a = rng.integers(size // 4, size // 2 - 2)
            b = rng.integers(size // 5, size // 3)
            ang = rng.uniform(0, math.pi)
            ca, sa = math.cos(ang), math.sin(ang)
            Xr = ca * (X - cx) + sa * (Y - cy)
            Yr = -sa * (X - cx) + ca * (Y - cy)
            leaf_mask = (Xr / a) ** 2 + (Yr / b) ** 2 <= 1

            # Base background
            img = np.full((size, size, 3), [0.08, 0.08, 0.08], dtype=np.float32)
            
            # Leaf fill with subtle noise
            leaf_noise = rng.normal(0, 0.04, (size, size, 3)).astype(np.float32)
            leaf_color = np.array([
                rng.uniform(base_r - 0.05, base_r + 0.05),
                rng.uniform(base_g - 0.05, base_g + 0.05),
                rng.uniform(base_b - 0.03, base_b + 0.03)
            ], dtype=np.float32)
            img[leaf_mask] = np.clip(leaf_color + leaf_noise[leaf_mask], 0.0, 1.0)

            # Disease spots/lesions if unhealthy
            if not is_healthy:
                num_spots = rng.integers(4, 12)
                for _ in range(num_spots):
                    sx = rng.integers(10, size - 10)
                    sy = rng.integers(10, size - 10)
                    sr = rng.integers(2, 6)
                    spot_mask = ((X - sx) ** 2 + (Y - sy) ** 2 <= sr ** 2) & leaf_mask
                    spot_color = np.array([
                        rng.uniform(0.65, 0.95),
                        rng.uniform(0.10, 0.35),
                        rng.uniform(0.02, 0.15)
                    ], dtype=np.float32)
                    img[spot_mask] = np.clip(spot_color + leaf_noise[spot_mask], 0.0, 1.0)

            imgs[i] = img

        X_all.append(imgs)
        y_all.append(np.full(samples_per_class, ci, dtype=np.int64))

    X_all = np.concatenate(X_all, axis=0)
    y_all = np.concatenate(y_all, axis=0)
    return X_all, y_all

SAMPLES_PER_CLASS = 80
X_raw, y_raw = generate_dataset(SAMPLES_PER_CLASS, IMG_SIZE)
perm = np.random.default_rng(123).permutation(len(X_raw))
X_raw, y_raw = X_raw[perm], y_raw[perm]

print(f"  Total samples : {len(X_raw)} ({SAMPLES_PER_CLASS} per class across {NUM_CLASSES} classes)", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# 3. PyTorch Dataset & DataLoaders
# ─────────────────────────────────────────────────────────────────────────────
class PlantDataset(Dataset):
    def __init__(self, images, labels, augment=False):
        # Transpose from (N, H, W, 3) to (N, 3, H, W)
        self.images = torch.from_numpy(images).permute(0, 3, 1, 2)
        self.labels = torch.from_numpy(labels)
        self.augment = augment

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.images[idx]
        lbl = self.labels[idx]
        if self.augment:
            if torch.rand(1).item() > 0.5:
                img = torch.flip(img, dims=[2])
            if torch.rand(1).item() > 0.5:
                img = torch.flip(img, dims=[1])
        return img, lbl

total_n = len(X_raw)
train_n = int(0.70 * total_n)
val_n   = int(0.15 * total_n)
test_n  = total_n - train_n - val_n

train_ds = PlantDataset(X_raw[:train_n], y_raw[:train_n], augment=True)
val_ds   = PlantDataset(X_raw[train_n:train_n+val_n], y_raw[train_n:train_n+val_n], augment=False)
test_ds  = PlantDataset(X_raw[train_n+val_n:], y_raw[train_n+val_n:], augment=False)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

print(f"  Train samples : {train_n} | Val: {val_n} | Test: {test_n}", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# 4. Vision Transformer (ViT) Architecture – 100% CNN-free
# ─────────────────────────────────────────────────────────────────────────────
print("\n[2/5] Constructing Vision Transformer (ViT) Model …", flush=True)

class PatchEmbedding(nn.Module):
    """Extracts patches and projects linearly WITHOUT any 2D convolution."""
    def __init__(self, img_size, patch_size, in_channels, embed_dim):
        super().__init__()
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        self.patch_dim = in_channels * patch_size * patch_size
        self.unfold = nn.Unfold(kernel_size=patch_size, stride=patch_size)
        self.linear_proj = nn.Linear(self.patch_dim, embed_dim)

    def forward(self, x):
        # x: (B, C, H, W)
        x = self.unfold(x)               # (B, patch_dim, n_patches)
        x = x.transpose(1, 2)           # (B, n_patches, patch_dim)
        return self.linear_proj(x)       # (B, n_patches, embed_dim)

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        out = (attn @ v).transpose(1, 2).reshape(B, N, D)
        return self.dropout(self.proj(out))

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=3, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class VisionTransformer(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, embed_dim,
                 num_heads, num_layers, mlp_ratio, dropout, num_classes):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        n_patches = (img_size // patch_size) ** 2
        
        # Learnable Class Token and Positional Embeddings
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.pos_embed = nn.Parameter(torch.randn(1, n_patches + 1, embed_dim) * 0.02)
        self.pos_dropout = nn.Dropout(dropout)

        self.blocks = nn.Sequential(*[
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes)
        )

    def forward(self, x):
        B = x.size(0)
        x = self.patch_embed(x)                         # (B, n_patches, embed_dim)
        cls = self.cls_token.expand(B, -1, -1)          # (B, 1, embed_dim)
        x = torch.cat([cls, x], dim=1)                  # (B, n_patches + 1, embed_dim)
        x = self.pos_dropout(x + self.pos_embed)
        x = self.blocks(x)
        x = self.norm(x)
        cls_out = x[:, 0]                               # Extract CLS token
        return self.head(cls_out)

model = VisionTransformer(
    img_size=IMG_SIZE,
    patch_size=PATCH_SIZE,
    in_channels=3,
    embed_dim=EMBED_DIM,
    num_heads=NUM_HEADS,
    num_layers=NUM_LAYERS,
    mlp_ratio=MLP_RATIO,
    dropout=DROPOUT,
    num_classes=NUM_CLASSES
).to(DEVICE)

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"  Total ViT Parameters : {total_params:,}", flush=True)
print(f"  Layers: {NUM_LAYERS} | Self-Attention Heads: {NUM_HEADS} | Embedding: {EMBED_DIM}", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# 5. Training Loop
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n[3/5] Training Vision Transformer for {EPOCHS} epochs …", flush=True)
criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)

history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
best_val_acc = 0.0
best_model_path = os.path.join(RESULTS, "best_vit_plant.pth")

print(f"  {'Epoch':>5}  {'TrLoss':>8}  {'TrAcc':>7}  {'VaLoss':>8}  {'VaAcc':>7}  {'Time':>6}", flush=True)
print("  " + "-" * 50, flush=True)

start_time = time.time()
for epoch in range(1, EPOCHS + 1):
    ep_start = time.time()
    
    # Train
    model.train()
    tr_loss, tr_correct, tr_total = 0.0, 0, 0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        tr_loss += loss.item() * imgs.size(0)
        tr_correct += (outputs.argmax(1) == labels).sum().item()
        tr_total += imgs.size(0)
        
    scheduler.step()
    
    # Validation
    model.eval()
    va_loss, va_correct, va_total = 0.0, 0, 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            va_loss += loss.item() * imgs.size(0)
            va_correct += (outputs.argmax(1) == labels).sum().item()
            va_total += imgs.size(0)
            
    epoch_tr_loss = tr_loss / tr_total
    epoch_tr_acc  = tr_correct / tr_total
    epoch_va_loss = va_loss / va_total
    epoch_va_acc  = va_correct / va_total
    
    history['train_loss'].append(epoch_tr_loss)
    history['train_acc'].append(epoch_tr_acc)
    history['val_loss'].append(epoch_va_loss)
    history['val_acc'].append(epoch_va_acc)
    
    ep_duration = time.time() - ep_start
    print(f"  {epoch:>5}  {epoch_tr_loss:>8.4f}  {epoch_tr_acc:>6.2%}  {epoch_va_loss:>8.4f}  {epoch_va_acc:>6.2%}  {ep_duration:>5.1f}s", flush=True)
    
    if epoch_va_acc > best_val_acc:
        best_val_acc = epoch_va_acc
        torch.save(model.state_dict(), best_model_path)

elapsed_total = time.time() - start_time
print(f"\n  Training finished in {elapsed_total:.1f}s (Best Val Acc: {best_val_acc:.2%})", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# 6. Evaluation on Test Set
# ─────────────────────────────────────────────────────────────────────────────
print("\n[4/5] Evaluating on Test Set …", flush=True)
model.load_state_dict(torch.load(best_model_path, map_location=DEVICE))
model.eval()

y_preds, y_trues = [], []
with torch.no_grad():
    for imgs, labels in test_loader:
        imgs = imgs.to(DEVICE)
        preds = model(imgs).argmax(dim=1).cpu().numpy()
        y_preds.extend(preds)
        y_trues.extend(labels.numpy())

y_preds = np.array(y_preds)
y_trues = np.array(y_trues)

test_acc = accuracy_score(y_trues, y_preds)
f1_macro = f1_score(y_trues, y_preds, average='macro', zero_division=0)
f1_weighted = f1_score(y_trues, y_preds, average='weighted', zero_division=0)

print(f"  Test Accuracy : {test_acc * 100:.2f}%", flush=True)
print(f"  Macro F1      : {f1_macro:.4f}", flush=True)
print(f"  Weighted F1   : {f1_weighted:.4f}", flush=True)
print("\nClassification Report:\n", classification_report(y_trues, y_preds, target_names=CLASS_NAMES, zero_division=0), flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# 7. Visualization Generation
# ─────────────────────────────────────────────────────────────────────────────
print("[5/5] Generating and saving visualization figures …", flush=True)

# 7a. Learning Curves
plt.figure(figsize=(12, 4.5))
plt.subplot(1, 2, 1)
plt.plot(range(1, EPOCHS + 1), history['train_acc'], 'o-', label='Train Acc', color='#2b5c8f')
plt.plot(range(1, EPOCHS + 1), history['val_acc'], 's--', label='Val Acc', color='#e06d53')
plt.title('ViT Model Accuracy over Epochs', fontweight='bold')
plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.legend(); plt.grid(True, alpha=0.3); plt.ylim(0, 1.05)

plt.subplot(1, 2, 2)
plt.plot(range(1, EPOCHS + 1), history['train_loss'], 'o-', label='Train Loss', color='#2b5c8f')
plt.plot(range(1, EPOCHS + 1), history['val_loss'], 's--', label='Val Loss', color='#e06d53')
plt.title('ViT Model Cross-Entropy Loss', fontweight='bold')
plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend(); plt.grid(True, alpha=0.3)

plt.tight_layout()
curve_file = os.path.join(RESULTS, "training_curves.png")
plt.savefig(curve_file, dpi=160)
plt.close()
print(f"  ✓ Saved: {curve_file}", flush=True)

# 7b. Confusion Matrix
cm = confusion_matrix(y_trues, y_preds)
plt.figure(figsize=(12, 10))
plt.imshow(cm, cmap='Blues')
plt.title("Confusion Matrix – ViT Plant Disease Classifier", fontweight='bold', fontsize=13)
plt.colorbar(fraction=0.046, pad=0.04)
ticks = np.arange(NUM_CLASSES)
labels_display = [n.replace("___", "\n") for n in CLASS_NAMES]
plt.xticks(ticks, labels_display, rotation=45, ha='right', fontsize=7.5)
plt.yticks(ticks, labels_display, fontsize=7.5)
plt.ylabel('True Class', fontweight='bold'); plt.xlabel('Predicted Class', fontweight='bold')

for i in range(NUM_CLASSES):
    for j in range(NUM_CLASSES):
        val = cm[i, j]
        plt.text(j, i, str(val), ha='center', va='center',
                 color='white' if val > cm.max()/2 else 'black', fontsize=8)

plt.tight_layout()
cm_file = os.path.join(RESULTS, "confusion_matrix.png")
plt.savefig(cm_file, dpi=160)
plt.close()
print(f"  ✓ Saved: {cm_file}", flush=True)

# 7c. Sample Predictions Grid
raw_test_imgs = X_raw[train_n+val_n:]
raw_test_lbls = y_raw[train_n+val_n:]
indices = np.random.default_rng(7).choice(len(raw_test_imgs), 16, replace=False)

fig, axes = plt.subplots(4, 4, figsize=(13, 13))
fig.suptitle("Sample Test Set Predictions (Vision Transformer)", fontsize=14, fontweight='bold')

for ax, idx in zip(axes.flatten(), indices):
    img = raw_test_imgs[idx]
    true_cls = CLASS_NAMES[raw_test_lbls[idx]]
    
    tensor_img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        out = torch.softmax(model(tensor_img), dim=1)[0].cpu().numpy()
    pred_idx = out.argmax()
    conf = out.max()
    pred_cls = CLASS_NAMES[pred_idx]
    
    correct = (true_cls == pred_cls)
    ax.imshow(img)
    color = "#1b8a36" if correct else "#c02828"
    ax.set_title(f"True: {true_cls.split('___')[1]}\nPred: {pred_cls.split('___')[1]} ({conf*100:.0f}%)",
                 fontsize=8, color=color, fontweight='bold')
    ax.axis('off')
    for spine in ax.spines.values():
        spine.set_edgecolor(color)
        spine.set_linewidth(2)

plt.tight_layout()
samples_file = os.path.join(RESULTS, "sample_predictions.png")
plt.savefig(samples_file, dpi=160)
plt.close()
print(f"  ✓ Saved: {samples_file}", flush=True)

# 7d. Per-Class F1 Score Chart
f1_per_class = f1_score(y_trues, y_preds, average=None, zero_division=0)
plt.figure(figsize=(13, 5.5))
bar_colors = ['#2ca02c' if s >= 0.75 else '#ff7f0e' if s >= 0.5 else '#d62728' for s in f1_per_class]
bars = plt.bar(range(NUM_CLASSES), f1_per_class, color=bar_colors, edgecolor='black', alpha=0.85)
plt.xticks(range(NUM_CLASSES), labels_display, rotation=45, ha='right', fontsize=7.5)
plt.ylabel('F1-Score', fontweight='bold')
plt.ylim(0, 1.05)
plt.title('Per-Class F1 Score Breakdown (ViT Classifier)', fontweight='bold', fontsize=12)
plt.axhline(f1_macro, color='darkblue', linestyle='--', linewidth=1.5, label=f'Macro F1: {f1_macro:.2f}')
plt.grid(axis='y', alpha=0.3)

legend_patches = [
    mpatches.Patch(color='#2ca02c', label='F1 ≥ 0.75'),
    mpatches.Patch(color='#ff7f0e', label='0.50 ≤ F1 < 0.75'),
    mpatches.Patch(color='#d62728', label='F1 < 0.50')
]
plt.legend(handles=legend_patches + [plt.Line2D([0], [0], color='darkblue', linestyle='--', label=f'Macro F1: {f1_macro:.2f}')], loc='lower right')
plt.tight_layout()
f1_file = os.path.join(RESULTS, "per_class_f1.png")
plt.savefig(f1_file, dpi=160)
plt.close()
print(f"  ✓ Saved: {f1_file}", flush=True)

# 7e. Architecture diagram
plt.figure(figsize=(11, 4.5))
ax = plt.gca()
ax.axis('off')
plt.title("Vision Transformer (ViT) – Zero Convolution Architecture", fontweight='bold', fontsize=12)

def draw_block(x, y, w, h, label, color):
    rect = plt.Rectangle((x, y), w, h, facecolor=color, edgecolor='black', linewidth=1.5, zorder=2)
    ax.add_patch(rect)
    ax.text(x + w/2, y + h/2, label, ha='center', va='center', fontweight='bold', fontsize=8, zorder=3)

def draw_arrow(x1, x2, y):
    ax.annotate("", xy=(x2, y), xytext=(x1, y), arrowprops=dict(arrowstyle="->", lw=1.5, color='black'))

blocks = [
    (0.05, 0.4, 0.12, 0.35, "Input Leaf\n96×96 (RGB)", "#AED6F1"),
    (0.22, 0.4, 0.13, 0.35, "Patch Split\n12×12\n(64 patches)", "#A9DFBF"),
    (0.39, 0.4, 0.13, 0.35, "Linear Proj\n+ [CLS] Token\n+ Pos Embed", "#FAD7A0"),
    (0.56, 0.4, 0.17, 0.35, f"×{NUM_LAYERS} Transformer\nEncoder Blocks\n(Multi-Head Attn)", "#D2B4DE"),
    (0.77, 0.4, 0.10, 0.35, "LayerNorm\n+ [CLS]", "#F9E79F"),
    (0.91, 0.4, 0.08, 0.35, f"MLP Head\n({NUM_CLASSES} cls)", "#F1948A")
]

for x, y, w, h, lbl, clr in blocks:
    draw_block(x, y, w, h, lbl, clr)

draw_arrow(0.17, 0.22, 0.57)
draw_arrow(0.35, 0.39, 0.57)
draw_arrow(0.52, 0.56, 0.57)
draw_arrow(0.73, 0.77, 0.57)
draw_arrow(0.87, 0.91, 0.57)

ax.text(0.5, 0.18, "⭐ No convolutional layers: uses pure multi-head self-attention and patch embeddings",
        ha='center', fontsize=9, style='italic', color='darkgreen', fontweight='bold')
plt.xlim(0, 1.02); plt.ylim(0, 0.9)
plt.tight_layout()
arch_file = os.path.join(RESULTS, "vit_architecture.png")
plt.savefig(arch_file, dpi=160)
plt.close()
print(f"  ✓ Saved: {arch_file}", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# 8. Summary
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65, flush=True)
print("  PROJECT EXECUTION FINISHED SUCCESSFULLY!", flush=True)
print("=" * 65, flush=True)
print(f"  Model Architecture : Vision Transformer (ViT) - 100% Non-CNN", flush=True)
print(f"  Trainable Params   : {total_params:,}", flush=True)
print(f"  Test Accuracy      : {test_acc * 100:.2f}%", flush=True)
print(f"  Macro F1-Score     : {f1_macro:.4f}", flush=True)
print(f"  Artifacts Saved To : {RESULTS}", flush=True)
print("=" * 65 + "\n", flush=True)
