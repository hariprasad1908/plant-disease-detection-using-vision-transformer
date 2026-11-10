# 🌿 Plant Leaf Disease Detection using Vision Transformer (ViT)

A Deep Learning project for multi-class plant disease classification using a **Vision Transformer (ViT)** implemented from scratch in **PyTorch**.

> **Note on Architecture:** This project contains **ZERO Convolutional Neural Network (CNN) layers** (`Conv2d`, `Conv1d`, etc.). Instead, it uses pure **Multi-Head Self-Attention (MHSA)**, patch linear projections (`nn.Unfold` + `nn.Linear`), learnable class tokens, and transformer encoder blocks.

---

## 🏗️ Vision Transformer Architecture

```
                      Input Leaf Image (96 × 96 × 3)
                                    │
                                    ▼
                      Patch Extraction (12 × 12 patches)
                              (64 patches total)
                                    │
                                    ▼
                      Linear Embedding Projection
                             (Dim: 432 → 128)
                                    │
                                    ▼
             Prepend Learnable [CLS] Token + 1D Positional Embeddings
                                    │
                                    ▼
       ┌───────────────────────────────────────────────────────────┐
       │         × 4 Transformer Encoder Blocks                     │
       │  ┌─────────────────────────────────────────────────────┐  │
       │  │ • Layer Normalization                               │  │
       │  │ • Multi-Head Self-Attention (4 Heads, Dim: 32/head) │  │
       │  │ • Residual Connection                               │  │
       │  │ • Layer Normalization                               │  │
       │  │ • Multi-Layer Perceptron (MLP with GELU & Dropout)  │  │
       │  │ • Residual Connection                               │  │
       │  └─────────────────────────────────────────────────────┘  │
       └───────────────────────────────────────────────────────────┘
                                    │
                                    ▼
                             Layer Normalization
                                    │
                                    ▼
                          Extract [CLS] Token (Dim 128)
                                    │
                                    ▼
                       MLP Classification Head
                                    │
                                    ▼
                     Softmax Probabilities (15 Classes)
```

---

## 📋 Disease Classes Covered (PlantVillage Taxonomy)

1. `Apple___Apple_scab`
2. `Apple___Black_rot`
3. `Apple___healthy`
4. `Corn___Common_rust`
5. `Corn___healthy`
6. `Grape___Black_rot`
7. `Grape___healthy`
8. `Potato___Early_blight`
9. `Potato___Late_blight`
10. `Potato___healthy`
11. `Tomato___Bacterial_spot`
12. `Tomato___Early_blight`
13. `Tomato___healthy`
14. `Tomato___Leaf_Mold`
15. `Tomato___Yellow_Leaf_Curl_Virus`

---

## 🚀 How to Run

### 1. Requirements
Install dependencies using pip:
```bash
pip install -r requirements.txt
```

### 2. Execute Training & Evaluation
Run the pipeline:
```bash
python plant_disease_vit.py
```

---

## 📊 Generated Output Artifacts (`results/`)

All training metrics and visualizations are automatically saved to `results/`:
- **`best_vit_plant.pth`**: Saved PyTorch model checkpoint.
- **`training_curves.png`**: Dual-plot graph showing Train & Validation Accuracy and Loss.
- **`confusion_matrix.png`**: 15×15 class confusion matrix heatmap.
- **`sample_predictions.png`**: Sample leaf image predictions with true vs. predicted labels.
- **`per_class_f1.png`**: Detailed F1-score performance breakdown per crop disease class.
- **`vit_architecture.png`**: Visual block diagram of the non-CNN transformer pipeline.

---

## 💡 Why Vision Transformer over CNN?
- **Global Receptive Field**: Unlike CNNs which look through small local kernels (e.g., $3 \times 3$), Vision Transformers model interactions across all patches of the image simultaneously from the very first layer.
- **Attention Interpretability**: Self-attention maps directly show which parts of the leaf (lesions, discolorations, margins) the model prioritized to make its diagnostic decision.
