

import os
import io
import math
import base64
import torch
import torch.nn as nn
from PIL import Image
from flask import Flask, render_template_string, request, jsonify

# ─────────────────────────────────────────────────────────────────────────────
# 1. Disease Information Database (Descriptions & Remedies)
# ─────────────────────────────────────────────────────────────────────────────
DISEASE_INFO = {
    "Apple___Apple_scab": {
        "crop": "Apple",
        "condition": "Apple Scab (Venturia inaequalis)",
        "status": "Diseased",
        "description": "A fungal disease causing olive-green to black velvety spots on leaves and fruit, leading to premature leaf drop and cracked fruit.",
        "treatment": "Rake and destroy fallen leaves in autumn. Apply fungicides like copper hydroxide or captan in early spring as buds break."
    },
    "Apple___Black_rot": {
        "crop": "Apple",
        "condition": "Black Rot (Diplodia seriata)",
        "status": "Diseased",
        "description": "Produces reddish-brown 'frog-eye' leaf spots, fruit rotting with concentric black rings, and bark cankers.",
        "treatment": "Prune out dead wood and mummified fruit. Apply sulfur or captan based sprays from petal fall through harvest."
    },
    "Apple___healthy": {
        "crop": "Apple",
        "condition": "Healthy Apple Leaf",
        "status": "Healthy",
        "description": "The leaf shows uniform green coloration, clear veins, and no necrotic lesions or fungal growth.",
        "treatment": "Maintain balanced irrigation, proper aeration via canopy pruning, and routine preventive inspection."
    },
    "Corn___Common_rust": {
        "crop": "Corn (Maize)",
        "condition": "Common Rust (Puccinia sorghi)",
        "status": "Diseased",
        "description": "Characterized by golden-brown to cinnamon-brown powdery pustules on both upper and lower leaf surfaces.",
        "treatment": "Plant rust-resistant corn hybrids. Apply foliar fungicides (e.g., azoxystrobin or pyraclostrobin) if infection starts early."
    },
    "Corn___healthy": {
        "crop": "Corn (Maize)",
        "condition": "Healthy Corn Leaf",
        "status": "Healthy",
        "description": "Vibrant elongated green blade with intact parallel venation and zero fungal pustules.",
        "treatment": "Ensure adequate nitrogen management and avoid dense planting to promote air circulation."
    },
    "Grape___Black_rot": {
        "crop": "Grape",
        "condition": "Black Rot (Guignardia bidwellii)",
        "status": "Diseased",
        "description": "Causes circular tan lesions with dark margins on leaves and black, shriveled, mummified berries.",
        "treatment": "Remove and burn mummified clusters. Apply myclobutanil or mancozeb starting at early shoot growth."
    },
    "Grape___healthy": {
        "crop": "Grape",
        "condition": "Healthy Grapevine Leaf",
        "status": "Healthy",
        "description": "Robust palmate leaf with deep green hue, healthy margins, and no fungal patches.",
        "treatment": "Maintain vine trellis spacing, ensure sunlight penetration, and monitor soil moisture."
    },
    "Potato___Early_blight": {
        "crop": "Potato",
        "condition": "Early Blight (Alternaria solani)",
        "status": "Diseased",
        "description": "Concentric rings forming a 'target board' pattern surrounded by a yellow chlorotic halo on older foliage.",
        "treatment": "Practice crop rotation with non-solanaceous crops. Apply chlorothalonil or copper-based fungicides regularly."
    },
    "Potato___Late_blight": {
        "crop": "Potato",
        "condition": "Late Blight (Phytophthora infestans)",
        "status": "Diseased",
        "description": "Water-soaked dark lesions rapidly expanding into white mildew on leaf undersides under humid conditions.",
        "treatment": "Destroy cull piles. Apply systemic fungicides like metalaxyl or cymoxanil immediately at first sign."
    },
    "Potato___healthy": {
        "crop": "Potato",
        "condition": "Healthy Potato Leaf",
        "status": "Healthy",
        "description": "Compound leaf displays vibrant foliage without necrotic blemishes or wilting.",
        "treatment": "Use certified disease-free seed tubers and hill up soil properly around stems."
    },
    "Tomato___Bacterial_spot": {
        "crop": "Tomato",
        "condition": "Bacterial Spot (Xanthomonas spp.)",
        "status": "Diseased",
        "description": "Small, dark, water-soaked circular lesions that dry up and turn brown, causing leaf yellowing and defoliation.",
        "treatment": "Avoid overhead sprinkler irrigation. Spray copper bactericides mixed with mancozeb for synergistic protection."
    },
    "Tomato___Early_blight": {
        "crop": "Tomato",
        "condition": "Early Blight (Alternaria solani)",
        "status": "Diseased",
        "description": "Dark brown target-shaped spots with yellow halos appearing first on the lower, oldest leaves.",
        "treatment": "Mulch around plants to prevent soil splashback. Prune lower foliage and apply copper fungicide."
    },
    "Tomato___healthy": {
        "crop": "Tomato",
        "condition": "Healthy Tomato Leaf",
        "status": "Healthy",
        "description": "Crisp green serrated leaflets free of fungal spores, bacterial speckling, or viral curling.",
        "treatment": "Keep consistent watering schedules, stake plants for vertical airflow, and practice companion planting."
    },
    "Tomato___Leaf_Mold": {
        "crop": "Tomato",
        "condition": "Leaf Mold (Passalora fulva)",
        "status": "Diseased",
        "description": "Pale greenish-yellow spots on upper leaf surfaces with olive-velvety fungal mold on corresponding undersides.",
        "treatment": "Lower relative humidity below 85% in greenhouses. Improve ventilation and apply copper or chlorothalonil."
    },
    "Tomato___Yellow_Leaf_Curl_Virus": {
        "crop": "Tomato",
        "condition": "Yellow Leaf Curl Virus (TYLCV)",
        "status": "Diseased",
        "description": "Severe upward leaf cupping, chlorotic leaf margins, bushy stunted growth, and heavy blossom drop.",
        "treatment": "Control silverleaf whitefly vectors using yellow sticky traps and insecticidal soap or neem oil. Remove infected plants."
    }
}

CLASS_NAMES = list(DISEASE_INFO.keys())
NUM_CLASSES = len(CLASS_NAMES)
IMG_SIZE = 96
PATCH_SIZE = 12
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─────────────────────────────────────────────────────────────────────────────
# 2. Vision Transformer (ViT) Architecture (100% Non-CNN)
# ─────────────────────────────────────────────────────────────────────────────
class PatchEmbedding(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, embed_dim):
        super().__init__()
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        self.patch_dim = in_channels * patch_size * patch_size
        self.unfold = nn.Unfold(kernel_size=patch_size, stride=patch_size)
        self.linear_proj = nn.Linear(self.patch_dim, embed_dim)

    def forward(self, x):
        x = self.unfold(x).transpose(1, 2)
        return self.linear_proj(x)

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
        attn = ((q @ k.transpose(-2, -1)) * self.scale).softmax(dim=-1)
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
    def __init__(self, img_size=96, patch_size=12, in_channels=3, embed_dim=128,
                 num_heads=4, num_layers=4, mlp_ratio=3, dropout=0.1, num_classes=15):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        n_patches = (img_size // patch_size) ** 2
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
        x = self.patch_embed(x)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = self.pos_dropout(x + self.pos_embed)
        x = self.blocks(x)
        x = self.norm(x)
        cls_out = x[:, 0]
        return self.head(cls_out)

# ─────────────────────────────────────────────────────────────────────────────
# 3. Load Trained Model Weights
# ─────────────────────────────────────────────────────────────────────────────
model = VisionTransformer().to(DEVICE)
weights_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "best_vit_plant.pth")

if os.path.exists(weights_path):
    model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
    print(f"✓ Successfully loaded trained ViT checkpoint: {weights_path}")
else:
    print(f"⚠ Checkpoint not found at {weights_path}, using initialized weights.")
model.eval()

# ─────────────────────────────────────────────────────────────────────────────
# 4. Prediction Helper
# ─────────────────────────────────────────────────────────────────────────────
def predict_leaf_image(pil_img):
    img = pil_img.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    arr = torch.tensor(list(img.getdata()), dtype=torch.float32).view(IMG_SIZE, IMG_SIZE, 3) / 255.0
    tensor = arr.permute(2, 0, 1).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

    top_indices = probs.argsort()[::-1]
    top1_idx = top_indices[0]
    top1_class = CLASS_NAMES[top1_idx]
    top1_conf = float(probs[top1_idx])

    top3 = []
    for idx in top_indices[:3]:
        cls_name = CLASS_NAMES[idx]
        info = DISEASE_INFO[cls_name]
        top3.append({
            "class_raw": cls_name,
            "crop": info["crop"],
            "condition": info["condition"],
            "status": info["status"],
            "prob": round(float(probs[idx]) * 100, 1)
        })

    primary_info = DISEASE_INFO[top1_class]
    return {
        "class_name": top1_class,
        "crop": primary_info["crop"],
        "condition": primary_info["condition"],
        "status": primary_info["status"],
        "confidence": round(top1_conf * 100, 1),
        "description": primary_info["description"],
        "treatment": primary_info["treatment"],
        "top3": top3
    }

# ─────────────────────────────────────────────────────────────────────────────
# 5. Flask Web Application & HTML Template
# ─────────────────────────────────────────────────────────────────────────────
app = Flask(__name__)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>🌿 Plant Disease Detection | Vision Transformer (ViT)</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap" rel="stylesheet">
  <style>
    :root {
      --primary: #15803d;
      --primary-light: #22c55e;
      --primary-dark: #166534;
      --accent: #eab308;
      --bg: #0f172a;
      --card-bg: #1e293b;
      --border: #334155;
      --text: #f8fafc;
      --text-muted: #94a3b8;
      --healthy: #10b981;
      --diseased: #ef4444;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: 'Plus Jakarta Sans', sans-serif;
      background: linear-gradient(135deg, #0b1120 0%, #0f172a 50%, #1e1b4b 100%);
      color: var(--text);
      min-height: 100vh;
      display: flex;
      flex-direction: column;
    }
    header {
      padding: 1.5rem 2rem;
      border-bottom: 1px solid rgba(255, 255, 255, 0.08);
      backdrop-filter: blur(12px);
      background: rgba(15, 23, 42, 0.7);
      position: sticky;
      top: 0;
      z-index: 50;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }
    .logo {
      display: flex;
      align-items: center;
      gap: 0.75rem;
      font-size: 1.35rem;
      font-weight: 800;
      background: linear-gradient(120deg, #4ade80, #38bdf8);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
    }
    .badge-tag {
      background: rgba(34, 197, 94, 0.15);
      border: 1px solid rgba(34, 197, 94, 0.35);
      color: #4ade80;
      font-size: 0.75rem;
      padding: 0.35rem 0.75rem;
      border-radius: 9999px;
      font-weight: 600;
    }
    main {
      flex: 1;
      max-width: 1100px;
      width: 100%;
      margin: 2rem auto;
      padding: 0 1.5rem;
      display: grid;
      grid-template-columns: 1fr 1.15fr;
      gap: 2rem;
    }
    @media (max-width: 860px) {
      main { grid-template-columns: 1fr; }
    }
    .card {
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 1.25rem;
      padding: 1.75rem;
      box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.4), 0 8px 10px -6px rgba(0, 0, 0, 0.3);
    }
    .card-title {
      font-size: 1.2rem;
      font-weight: 700;
      margin-bottom: 0.5rem;
      display: flex;
      align-items: center;
      gap: 0.5rem;
    }
    .card-subtitle {
      color: var(--text-muted);
      font-size: 0.85rem;
      margin-bottom: 1.5rem;
    }
    .dropzone {
      border: 2px dashed #475569;
      border-radius: 1rem;
      padding: 2.5rem 1.5rem;
      text-align: center;
      background: rgba(15, 23, 42, 0.5);
      cursor: pointer;
      transition: all 0.25s ease;
      position: relative;
    }
    .dropzone:hover, .dropzone.dragover {
      border-color: var(--primary-light);
      background: rgba(34, 197, 94, 0.05);
    }
    .dropzone input[type="file"] {
      position: absolute;
      inset: 0;
      width: 100%;
      height: 100%;
      opacity: 0;
      cursor: pointer;
    }
    .drop-icon {
      font-size: 2.75rem;
      margin-bottom: 0.75rem;
      display: block;
    }
    .preview-container {
      margin-top: 1.25rem;
      display: none;
      text-align: center;
    }
    .preview-img {
      max-width: 100%;
      max-height: 240px;
      border-radius: 0.85rem;
      border: 2px solid var(--border);
      object-fit: cover;
    }
    .btn-diagnose {
      width: 100%;
      margin-top: 1.25rem;
      background: linear-gradient(135deg, var(--primary) 0%, var(--primary-light) 100%);
      color: white;
      border: none;
      padding: 0.9rem;
      border-radius: 0.75rem;
      font-size: 1rem;
      font-weight: 700;
      cursor: pointer;
      display: flex;
      justify-content: center;
      align-items: center;
      gap: 0.5rem;
      transition: all 0.2s ease;
    }
    .btn-diagnose:hover {
      box-shadow: 0 10px 20px -5px rgba(34, 197, 94, 0.4);
      transform: translateY(-1px);
    }
    .btn-diagnose:disabled {
      background: #475569;
      cursor: not-allowed;
      transform: none;
      box-shadow: none;
    }
    .spinner {
      display: none;
      width: 20px;
      height: 20px;
      border: 3px solid rgba(255,255,255,0.3);
      border-radius: 50%;
      border-top-color: white;
      animation: spin 0.8s linear infinite;
    }
    @keyframes spin { to { transform: rotate(360deg); } }
    
    .placeholder-state {
      height: 100%;
      min-height: 350px;
      display: flex;
      flex-direction: column;
      justify-content: center;
      align-items: center;
      text-align: center;
      color: var(--text-muted);
      padding: 2rem;
    }
    .placeholder-icon {
      font-size: 3.5rem;
      opacity: 0.35;
      margin-bottom: 1rem;
    }
    .result-container {
      display: none;
      animation: fadeIn 0.3s ease;
    }
    @keyframes fadeIn { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: translateY(0); } }
    .status-badge {
      display: inline-flex;
      align-items: center;
      gap: 0.4rem;
      padding: 0.4rem 0.9rem;
      border-radius: 9999px;
      font-size: 0.85rem;
      font-weight: 700;
      margin-bottom: 1rem;
    }
    .status-healthy { background: rgba(16, 185, 129, 0.15); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.3); }
    .status-diseased { background: rgba(239, 68, 68, 0.15); color: #f87171; border: 1px solid rgba(239, 68, 68, 0.3); }
    .disease-name {
      font-size: 1.5rem;
      font-weight: 800;
      margin-bottom: 0.25rem;
    }
    .crop-name {
      color: #38bdf8;
      font-size: 0.95rem;
      font-weight: 600;
      margin-bottom: 1.25rem;
    }
    .confidence-meter {
      background: #0f172a;
      border-radius: 0.65rem;
      padding: 0.85rem 1rem;
      margin-bottom: 1.5rem;
      border: 1px solid var(--border);
    }
    .conf-header {
      display: flex;
      justify-content: space-between;
      font-size: 0.85rem;
      font-weight: 600;
      margin-bottom: 0.5rem;
    }
    .progress-bar-bg {
      background: #334155;
      height: 8px;
      border-radius: 9999px;
      overflow: hidden;
    }
    .progress-bar-fill {
      background: linear-gradient(90deg, #22c55e, #38bdf8);
      height: 100%;
      border-radius: 9999px;
      width: 0%;
      transition: width 0.6s ease;
    }
    .info-section {
      background: rgba(15, 23, 42, 0.5);
      border-radius: 0.85rem;
      padding: 1.15rem;
      margin-bottom: 1rem;
      border: 1px solid rgba(255, 255, 255, 0.05);
    }
    .info-title {
      font-size: 0.85rem;
      font-weight: 700;
      color: #94a3b8;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      margin-bottom: 0.4rem;
      display: flex;
      align-items: center;
      gap: 0.4rem;
    }
    .info-text {
      font-size: 0.9rem;
      line-height: 1.5;
      color: #e2e8f0;
    }
    .top3-box {
      margin-top: 1.25rem;
    }
    .top3-title {
      font-size: 0.85rem;
      font-weight: 700;
      color: var(--text-muted);
      margin-bottom: 0.65rem;
    }
    .top3-item {
      display: flex;
      justify-content: space-between;
      font-size: 0.8rem;
      padding: 0.4rem 0.6rem;
      border-radius: 0.4rem;
      background: rgba(15, 23, 42, 0.4);
      margin-bottom: 0.35rem;
    }
    footer {
      text-align: center;
      padding: 1.5rem;
      font-size: 0.8rem;
      color: var(--text-muted);
      border-top: 1px solid rgba(255, 255, 255, 0.05);
      margin-top: auto;
    }
  </style>
</head>
<body>
  <header>
    <div class="logo">
      <span>🍃</span> PlantPulse AI
    </div>
    <div class="badge-tag">⚡ Vision Transformer (100% Non-CNN)</div>
  </header>

  <main>
    <!-- Left Column: Upload -->
    <div class="card">
      <h2 class="card-title">📷 Upload Plant Leaf</h2>
      <p class="card-subtitle">Upload or drag a clear image of an affected or healthy crop leaf</p>

      <div class="dropzone" id="dropzone">
        <input type="file" id="fileInput" accept="image/*">
        <span class="drop-icon">📤</span>
        <strong style="display:block; font-size:1rem; margin-bottom:0.25rem;">Drag & drop leaf photo here</strong>
        <span style="color:var(--text-muted); font-size:0.85rem;">or click to browse from device (JPG, PNG)</span>
      </div>

      <div class="preview-container" id="previewContainer">
        <img id="previewImg" class="preview-img" alt="Leaf Preview">
      </div>

      <button class="btn-diagnose" id="diagnoseBtn" disabled>
        <span class="spinner" id="spinner"></span>
        <span id="btnText">Analyze with Vision Transformer</span>
      </button>

      <div style="margin-top:1.5rem; padding-top:1.25rem; border-top:1px solid rgba(255,255,255,0.08);">
        <span style="font-size:0.8rem; font-weight:600; color:var(--text-muted);">Supported Plants:</span>
        <div style="display:flex; flex-wrap:wrap; gap:0.4rem; margin-top:0.5rem;">
          <span class="badge-tag" style="background:#1e293b; border-color:#475569; color:#cbd5e1;">🍎 Apple</span>
          <span class="badge-tag" style="background:#1e293b; border-color:#475569; color:#cbd5e1;">🌽 Corn</span>
          <span class="badge-tag" style="background:#1e293b; border-color:#475569; color:#cbd5e1;">🍇 Grape</span>
          <span class="badge-tag" style="background:#1e293b; border-color:#475569; color:#cbd5e1;">🥔 Potato</span>
          <span class="badge-tag" style="background:#1e293b; border-color:#475569; color:#cbd5e1;">🍅 Tomato</span>
        </div>
      </div>
    </div>

    <!-- Right Column: Results -->
    <div class="card">
      <div class="placeholder-state" id="placeholder">
        <div class="placeholder-icon">🔬</div>
        <h3 style="font-size:1.15rem; margin-bottom:0.5rem; color:#cbd5e1;">No Leaf Image Analyzed</h3>
        <p style="font-size:0.85rem; max-width:320px;">Upload a leaf photo on the left panel to run deep self-attention diagnosis and view remedies.</p>
      </div>

      <div class="result-container" id="resultContainer">
        <div id="statusBadge" class="status-badge"></div>
        <h2 class="disease-name" id="conditionTitle"></h2>
        <div class="crop-name" id="cropSubtitle"></div>

        <div class="confidence-meter">
          <div class="conf-header">
            <span>Model Confidence</span>
            <span id="confidenceValue" style="color:#38bdf8;"></span>
          </div>
          <div class="progress-bar-bg">
            <div class="progress-bar-fill" id="confidenceBar"></div>
          </div>
        </div>

        <div class="info-section">
          <div class="info-title">📖 Disease Description</div>
          <p class="info-text" id="descText"></p>
        </div>

        <div class="info-section">
          <div class="info-title">💊 Recommended Treatment & Action</div>
          <p class="info-text" id="treatmentText"></p>
        </div>

        <div class="top3-box">
          <div class="top3-title">Top Candidate Predictions:</div>
          <div id="top3List"></div>
        </div>
      </div>
    </div>
  </main>

  <footer>
    Powered by PyTorch <strong>Vision Transformer (ViT)</strong> • 4 Attention Heads • 64 Patches • Zero Convolution Layers
  </footer>

  <script>
    const fileInput = document.getElementById('fileInput');
    const dropzone = document.getElementById('dropzone');
    const previewContainer = document.getElementById('previewContainer');
    const previewImg = document.getElementById('previewImg');
    const diagnoseBtn = document.getElementById('diagnoseBtn');
    const btnText = document.getElementById('btnText');
    const spinner = document.getElementById('spinner');

    const placeholder = document.getElementById('placeholder');
    const resultContainer = document.getElementById('resultContainer');
    const statusBadge = document.getElementById('statusBadge');
    const conditionTitle = document.getElementById('conditionTitle');
    const cropSubtitle = document.getElementById('cropSubtitle');
    const confidenceValue = document.getElementById('confidenceValue');
    const confidenceBar = document.getElementById('confidenceBar');
    const descText = document.getElementById('descText');
    const treatmentText = document.getElementById('treatmentText');
    const top3List = document.getElementById('top3List');

    let currentFile = null;

    fileInput.addEventListener('change', (e) => {
      if (e.target.files && e.target.files[0]) {
        handleFile(e.target.files[0]);
      }
    });

    ['dragenter', 'dragover'].forEach(name => {
      dropzone.addEventListener(name, (e) => { e.preventDefault(); dropzone.classList.add('dragover'); });
    });
    ['dragleave', 'drop'].forEach(name => {
      dropzone.addEventListener(name, (e) => { e.preventDefault(); dropzone.classList.remove('dragover'); });
    });
    dropzone.addEventListener('drop', (e) => {
      if (e.dataTransfer.files && e.dataTransfer.files[0]) {
        handleFile(e.dataTransfer.files[0]);
      }
    });

    function handleFile(file) {
      currentFile = file;
      const reader = new FileReader();
      reader.onload = (e) => {
        previewImg.src = e.target.result;
        previewContainer.style.display = 'block';
        diagnoseBtn.disabled = false;
      };
      reader.readAsDataURL(file);
    }

    diagnoseBtn.addEventListener('click', async () => {
      if (!currentFile) return;

      diagnoseBtn.disabled = true;
      spinner.style.display = 'inline-block';
      btnText.innerText = 'Analyzing with Attention Heads…';

      const formData = new FormData();
      formData.append('file', currentFile);

      try {
        const response = await fetch('/predict', {
          method: 'POST',
          body: formData
        });
        const data = await response.json();

        if (data.error) {
          alert('Error: ' + data.error);
          return;
        }

        renderResults(data);
      } catch (err) {
        alert('Failed to connect to ViT model server: ' + err);
      } finally {
        diagnoseBtn.disabled = false;
        spinner.style.display = 'none';
        btnText.innerText = 'Analyze Another Leaf';
      }
    });

    function renderResults(data) {
      placeholder.style.display = 'none';
      resultContainer.style.display = 'block';

      if (data.status === 'Healthy') {
        statusBadge.className = 'status-badge status-healthy';
        statusBadge.innerHTML = '● Plant Leaf is Healthy';
      } else {
        statusBadge.className = 'status-badge status-diseased';
        statusBadge.innerHTML = '▲ Disease Detected';
      }

      conditionTitle.innerText = data.condition;
      cropSubtitle.innerText = 'Crop Family: ' + data.crop;
      confidenceValue.innerText = data.confidence + '%';
      confidenceBar.style.width = data.confidence + '%';

      descText.innerText = data.description;
      treatmentText.innerText = data.treatment;

      top3List.innerHTML = '';
      data.top3.forEach(item => {
        const row = document.createElement('div');
        row.className = 'top3-item';
        row.innerHTML = `<span><strong>${item.crop}</strong> – ${item.condition}</span><span style="font-weight:700; color:#38bdf8;">${item.prob}%</span>`;
        top3List.appendChild(row);
      });
    }
  </script>
</body>
</html>
"""

@app.route("/", methods=["GET"])
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400
    try:
        pil_img = Image.open(file.stream)
        results = predict_leaf_image(pil_img)
        return jsonify(results)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = 5000
    print("\n" + "=" * 65)
    print(f"  🌿 Plant Disease Detection Web App (Vision Transformer)")
    print(f"  ⚡ Running locally on: http://127.0.0.1:{port}")
    print(f"  ⚡ Or in your browser : http://localhost:{port}")
    print("=" * 65 + "\n")
    app.run(host="0.0.0.0", port=port, debug=False)
