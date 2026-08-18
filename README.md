# CDCIR — CLIP–DINOv2 for Composed Image Retrieval

[![Demo on Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Demo-Hugging%20Face%20Space-blue)](https://huggingface.co/spaces/chadhurbala/CDCIR)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-ee4c2c)](https://pytorch.org/)

**Composed Image Retrieval (CIR)** takes a reference image plus a text modification — *"is darker colored and red with black lettering"* — and must return, from a large gallery, the target image that satisfies the described change.

CDCIR does this with **both backbones frozen** and only **7.22 M trainable parameters**, reaching **R@10 = 40.22% / R@50 = 61.83%** on FashionIQ — within 1.5 points of a fully fine-tuned 211 M-parameter CLIP4CIR, and ahead of recent zero-shot textual-inversion and diffusion methods.

👉 **[Try the live demo](https://huggingface.co/spaces/chadhurbala/CDCIR)** — upload a reference image, type a modification, get the top-K matches.

---

## The problem with CLIP-only CIR

Most CIR systems use CLIP as the visual backbone. CLIP is pre-trained to align a *whole image* with a *whole caption*, so its global embedding is excellent but its **patch features are spatially weak**. This works for global edits (*"make it red"*, *"add long sleeves"*) and breaks on local ones (*"yellow collar"*, *"smaller logo"*, *"writing on the chest"*) — the model cannot tell *where* in the image the edit applies.

The two common workarounds both have costs:

- **Textual inversion** (Pic2Word, LinCIR, Context-I2W) compresses the reference image into one or a few pseudo-word tokens — fine-grained spatial detail is lost.
- **LLM / diffusion pipelines** (CIReVL, CompoDiff, OSrCIR) reason over captions or synthesise pseudo-targets — accurate, but multiple LLM calls or denoising steps per query make real-time retrieval impractical.

**DINOv2** is trained with self-distillation plus masked image modelling, so its patch embeddings *are* spatially coherent — good enough to drive segmentation and dense correspondence without supervision. That is exactly what the grounding step of CIR needs. CDCIR keeps CLIP's text-aligned retrieval space and borrows DINOv2's spatial precision.

---

## Method

![pipeline](https://img.shields.io/badge/-CLIP%20ViT--B%2F16%20%2B%20DINOv2%20ViT--S%2F14%20(both%20frozen)-lightgrey)

```
reference image ──┬── CLIP ViT-B/16 (frozen) ──► z_img  [512]      global anchor
                  └── DINOv2 ViT-S/14 (frozen) ─► P    [256, 384]  spatial patches
                                                        │
modification text ─── CLIP text encoder (frozen) ─┬─► Z_txt [77, 512] ──┐
                                                  └─► z_txt [512]       │
                                                                        ▼
                                    TQDKV cross-attention  (Q = text, K/V = DINO patches)
                                                    │
                                       learned-query attention pooler
                                                    │
                                                    ▼  p [512]
                            sigmoid-gated three-way fusion of [z_img ; z_txt ; p]
                                                    │
                                                    ▼
                                        composed query q [512]  ──► cosine similarity
                                                                     against gallery
```

### 1. Frozen backbones

| Backbone | Checkpoint | What we take |
|---|---|---|
| CLIP ViT-B/16 | OpenCLIP `laion2b_s34b_b88k` | image `[CLS]` → `z_img ∈ ℝ⁵¹²`; text `[EOS]` → `z_txt ∈ ℝ⁵¹²`; all 77 pre-pooling token features → `Z_txt ∈ ℝ⁷⁷ˣ⁵¹²` |
| DINOv2 ViT-S/14 | `facebookresearch/dinov2` | patch tokens `P ∈ ℝ²⁵⁶ˣ³⁸⁴` (a 16×16 grid) |

Neither backbone is ever back-propagated through.

### 2. TQDKV — Text-Query DINO-Key-Value cross-attention

A transformer block with two twists: **queries come from the text tokens, keys and values from the DINOv2 patches**, and the attention residual is modulated by a learnable scalar gate.

```
H       = LN(Z_txt)
A       = MHA(Q = H, K = P, V = P)          # 8 heads, q_dim 512, kv_dim 384
Z_txt  ← Z_txt + tanh(g) · A                # g initialised to 0.65
Z_txt  ← Z_txt + FFN(LN(Z_txt))             # GELU MLP, hidden 4 × 512
```

The `tanh` gate lets the model start close to the raw text tokens and inject visual information only as the attention becomes meaningful, which stabilises the early epochs. The 77 enriched tokens are then compressed into a single **grounded vector** `p ∈ ℝ⁵¹²` by a learned-query attention pooler.

> **One layer beats deeper stacks.** N = 2 and N = 3 both overfit on FashionIQ. We attribute this to dataset and batch size.

### 3. Sigmoid-gated three-way fusion

`p` knows *where* the edit applies but carries nothing about what should be **preserved** from the reference — and it degrades early in training while the cross-attention is still learning. So it is fused with the two global CLIP vectors:

```
c = [ z_img ; z_txt ; p ]  ∈ ℝ¹⁵³⁶
u = LN(MLP_fuse(c))                     # 2-layer GELU MLP → ℝ⁵¹²
λ = σ(MLP_gate(c))                      # 2-layer GELU MLP → ℝ, final bias = −1.0
q = (1 − λ) · u  +  λ · LN(z_txt)
```

The final bias of `−1.0` puts `λ ≈ 0.27` at initialisation, so `q` starts out dominated by a plain text embedding — a stable shortcut that the fusion path gradually improves upon.

### 4. Training objective

Symmetric InfoNCE on the composed query, **plus an auxiliary contrastive loss on the pooled vector**:

```
L = L_SymNCE(q, t, s)  +  α · L_SymNCE(p, t, s)
```

with `α = 2.0` for epochs 0–10, `1.0` for 10–25, `0.2` thereafter.

The auxiliary term is not a regulariser. Without it, gradients reach the cross-attention only through the fusion MLP — which is free to route around the attention path entirely and lean on the text shortcut. This loss creates a direct gradient path that forces TQDKV to learn real local features.

---

## Results (FashionIQ validation)

### vs. CLIP4CIR variants

| Model | Train. params | Dresses R@10 / R@50 | Shirts R@10 / R@50 | Tops&Tees R@10 / R@50 | Avg R@10 | Avg R@50 | **Avg** |
|---|---|---|---|---|---|---|---|
| CLIP4CIR (frozen) | 33 M | 27.22 / 50.62 | 31.85 / 52.50 | 33.81 / 57.57 | 30.96 | 53.56 | 42.26 |
| CLIP4CIR (image FT) | 120 M | 32.47 / 55.81 | 34.30 / 55.79 | 38.45 / 62.36 | 35.07 | 57.78 | 46.42 |
| CLIP4CIR (text FT) | 124 M | 31.43 / 54.98 | 35.87 / 57.21 | 38.20 / 63.22 | 35.16 | 58.47 | 46.81 |
| CLIP4CIR (full FT) | 211 M | 37.67 / 63.16 | 39.87 / 60.84 | 44.88 / 68.59 | 40.80 | 64.20 | 52.50 |
| **CDCIR (ours)** | **7.22 M** | 37.48 / 60.24 | 38.47 / 59.22 | 44.72 / 66.04 | **40.22** | **61.83** | **51.03** |

CDCIR lands **1.47 points** behind the fully fine-tuned 211 M model while training **~30× fewer parameters**, and beats the 33 M frozen baseline by **8.77 points**.

### vs. textual-inversion and diffusion methods

| Model | Backbone | Avg R@10 | Avg R@50 | **Avg** |
|---|---|---|---|---|
| Context-I2W | CLIP-L | 27.80 | 48.90 | 38.40 |
| CompoDiff | CLIP-L | 37.36 | 50.85 | 44.11 |
| LinCIR | CLIP-H | 36.26 | 57.46 | 46.86 |
| **CDCIR (ours)** | **CLIP-B + DINOv2** | **40.22** | **61.83** | **51.03** |

Best overall by **+4.17 to +12.63 points**, and highest R@10 in *every* category — using a **smaller** CLIP backbone than either LinCIR (CLIP-H) or Context-I2W (CLIP-L), and with no iterative denoising at query time.

### Ablation (R@10, features masked at inference)

| Configuration | R@10 | Δ |
|---|---|---|
| Full model | **40.22** | — |
| Masked CLIP image `[CLS]` | 35.80 | −4.42 |
| DINO patches → their mean | 35.49 | −4.73 |
| DINO patches shuffled | 35.35 | −4.87 |
| DINO patches zeroed | 33.24 | −6.98 |
| **Pooled vector `p` zeroed** | **27.32** | **−12.90** |

Each DINOv2 ablation costs ~5 points, but removing the pooled TQDKV pathway costs almost **13**. The grounded cross-attention path — and the auxiliary loss that trains it — is the single biggest reason for the gain. The text shortcut alone is not enough.

Attention heatmaps confirm this qualitatively: given a black t-shirt printed with *"got mandocellos?"* and the query *"is light blue with yellow logo"*, attention concentrates precisely over the printed-text region.

---

## Repository structure

```
.
├── CDCIR/
│   ├── app.py                  # Gradio demo — loads models, builds gallery, serves UI
│   ├── models.py               # TextQueryDINOLayer (TQDKV) + TextQueryDINOCombiner
│   ├── feature_extraction.py   # TargetPad transforms, CLIP 77-token extraction, DINOv2 patches
│   ├── retrieval.py            # Gallery class — L2-normalized index + cosine top-K
│   ├── eval_bundle.pt          # Trained combiner weights + training config
│   ├── val_clip_cls.pt         # Precomputed CLIP [CLS] features for the val gallery
│   └── requirements.txt
├── model.ipynb                 # Full training / evaluation / ablation notebook
└── README.md
```

> `CDCIR/val_images/` (8,167 FashionIQ validation images, ~67 MB) is **not** committed — see below.

---

## Running the demo locally

```bash
git clone https://github.com/amankoli2002/CDCIR-CLIP-DINOv2-for-Composed-Image-Retrieval.git
cd CDCIR-CLIP-DINOv2-for-Composed-Image-Retrieval/CDCIR
pip install -r requirements.txt
```

The gallery images are not in this repo (they belong to the FashionIQ dataset). Download them from the [FashionIQ repository](https://github.com/XiaoxiaoGuo/fashion-iq) and place the validation images under `CDCIR/val_images/`:

```
CDCIR/val_images/
├── 00/
├── 01/
├── 02/
└── 03/
```

Filenames must be the FashionIQ image IDs (e.g. `B0070TYTWO.jpg`) — `retrieval.py` indexes them by file stem and searches subfolders recursively. Then:

```bash
python app.py
```

`app.py` will download the CLIP and DINOv2 weights on first run (DINOv2 via `torch.hub` into `/tmp/torch_hub`) and print the epoch and R@10 stored in `eval_bundle.pt`. It runs on GPU if one is available and falls back to CPU otherwise. Set a custom data directory with `DATA_DIR=/path/to/files python app.py`.

**No setup needed?** The [hosted Space](https://huggingface.co/spaces/chadhurbala/CDCIR) already has the gallery indexed.

---

## Training details

| | |
|---|---|
| Dataset | FashionIQ — ~77 k images, 3 categories (dresses, shirts, tops & tees) |
| Training triplets | 18 k → **54 k** (both relative captions plus their concatenation) |
| Evaluation | concatenated caption only, standard FashionIQ protocol |
| Optimiser | AdamW, base LR 1e-5, weight decay 2e-5 |
| Schedule | OneCycleLR cosine, 20% warm-up, peak/start 20, peak/end 1000 |
| Special LR | 5× for gate parameters and the learnable temperature |
| Batch size | **256** — essential, see below |
| Epochs | 150 (converges around 50–60) |
| EMA | decay 0.998, used for evaluation |
| Regularisation | dropout 0.05, gradient-norm clip 1.0 |
| Temperature | learnable, initialised to `log(1/0.07)` |
| Preprocessing | white-pad to square × 1.25, resize to 224, centre-crop, per-backbone normalisation |
| Hardware | a single **NVIDIA T4** (Google Colab) |

### Things we learned the hard way

- **Symmetric InfoNCE needs large batches.** Under resource limits we tried fine-tuning the backbones with small batches; it consistently underperformed. Batch 256 with *frozen* backbones was strictly better than any small-batch fine-tuning run we attempted — contrastive losses need enough in-batch negatives.
- **One TQDKV layer is enough.** Two and three layers both overfit.
- **The auxiliary loss is necessary.** A minimal version — single cross-attention layer, no fusion MLP, composed query = `p` — plateaus around 30% R@10. The attention path produces useful features but cannot compose them with the global image and text on its own.
- **Hybrid gallery embeddings were unstable.** Concatenating or fusing DINOv2 into the *gallery* representation destabilised training, so the retrieval target stays the plain CLIP `[CLS]` embedding.

---

## Authors

Course project for **CS776 — Deep Learning for Computer Vision**, Department of CSE, Indian Institute of Technology Kanpur.

| Author | Contribution |
|---|---|
| Chadhurbala R V | core architecture, loss design, training, hyperparameter tuning, demo, ablation |
| **Aman Koli** | **ablation study, training, fusion-layer hyperparameter tuning, plots & visualisation** |
| Md Mahfooz Ansari | ablation study, plots, training, optimiser/scheduler tuning |
| Nihal Chourasiya | plots & visualisation, training, demo tool, fusion-layer tuning |
| Tarun Ghutey | optimiser/scheduler tuning, training, ablation study, demo tool |
| Ujjwal Kajal | training, transformer-layer tuning, plots, demo tool |
| Utkarsh Agrawal | ablation study, plots, fusion-layer tuning, training |

---

## References

1. Vo et al., *Composing Text and Image for Image Retrieval — An Empirical Odyssey*, CVPR 2019.
2. Baldrati et al., *Composed Image Retrieval using Contrastive Learning and Task-oriented CLIP-based Features* (CLIP4CIR), ACM TOMM 2023.
3. Saito et al., *Pic2Word: Mapping Pictures to Words for Zero-shot Composed Image Retrieval*, CVPR 2023.
4. Gu et al., *Language-only Training of Zero-shot Composed Image Retrieval* (LinCIR), CVPR 2024.
5. Tang et al., *Context-I2W: Mapping Images to Context-Dependent Words*, AAAI 2024.
6. Oquab et al., *DINOv2: Learning Robust Visual Features without Supervision*, TMLR 2024.
7. Radford et al., *Learning Transferable Visual Models From Natural Language Supervision* (CLIP), ICML 2021.

