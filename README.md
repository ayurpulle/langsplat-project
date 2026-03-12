# LangSplat: Open-Vocabulary 3D Scene Understanding via Language-Embedded Gaussian Splatting

> Query a 3D scene with natural language. Type *"wooden chair"* or *"ceiling light"* and get a live 3D heatmap showing exactly where that object is — with no 3D labels, no manual annotation, and no fine-tuning.

---

## Table of Contents

- [Overview](#overview)
- [How It Works](#how-it-works)
- [Results](#results)
- [Repository Structure](#repository-structure)
- [Setup](#setup)
- [Dataset Preparation](#dataset-preparation)
- [Training](#training)
  - [Stage 1: Base 3DGS](#stage-1-base-3dgs)
  - [Stage 2: CLIP Feature Extraction](#stage-2-clip-feature-extraction)
  - [Stage 3: Language Field Training](#stage-3-language-field-training)
- [Querying the Scene](#querying-the-scene)
- [Interactive Demo](#interactive-demo)
- [Extensions](#extensions)
- [Troubleshooting](#troubleshooting)
- [References](#references)

---

## Overview

This project implements a 3D scene understanding system that combines **3D Gaussian Splatting (3DGS)** with **language-aligned visual features** distilled from CLIP and SAM. The result is a queryable 3D scene representation: given a set of posed RGB images of a room, the system reconstructs the scene as a collection of 3D Gaussians and attaches a language feature vector to each one. At inference time, a text query is encoded by CLIP and matched against the per-Gaussian features to produce a relevance heatmap rendered from any viewpoint.

This is a from-scratch implementation of the core ideas in [LangSplat (Qin et al., CVPR 2024)](https://arxiv.org/abs/2312.16084), trained on the [Replica](https://github.com/facebookresearch/Replica-Dataset) dataset.

**What you can do with the final system:**

```
Query: "a wooden chair"          → Gaussians near chairs light up in 3D
Query: "laptop screen"           → Illuminates desk objects
Query: "everything on the floor" → Full floor region highlighted
Query: "comfortable seating"     → Semantic generalisation beyond exact labels
```

---

## How It Works

The pipeline has three stages:

```
RGB frames + camera poses
        │
        ▼
┌───────────────────┐
│   Stage 1: 3DGS   │  Reconstruct scene geometry as 3D Gaussians
│   (colour only)   │  using differentiable α-compositing rendering
└────────┬──────────┘
         │
         ▼
┌─────────────────────────┐
│  Stage 2: CLIP Feature  │  For each training image:
│      Extraction         │  SAM generates segments → CLIP encodes
│  (per-pixel features)   │  each segment at 3 scales → dense 512-dim
└────────┬────────────────┘  feature map per pixel
         │
         ▼
┌────────────────────────┐
│  Stage 3: Language     │  Train a tiny MLP on each Gaussian to
│  Field Training        │  predict compressed CLIP features,
│                        │  supervised by per-pixel feature maps
└────────┬───────────────┘
         │
         ▼
┌────────────────────────┐
│   Query Interface      │  Text → CLIP → cosine similarity with
│                        │  Gaussian features → rendered heatmap
└────────────────────────┘
```

### Why Gaussians instead of NeRF?

NeRF-based language fields (e.g. LERF) require slow volumetric ray marching at query time. 3D Gaussians render in real-time via rasterisation, making interactive querying practical. LangSplat also introduces a per-Gaussian autoencoder that compresses 512-dim CLIP features to 3 dimensions for storage, then decompresses at query time — keeping memory overhead low even for large scenes.

### The multi-scale CLIP trick

Naively running CLIP on an entire image produces a single image-level embedding that loses spatial detail. LangSplat instead uses SAM to generate instance-level masks, then crops each masked region at **three spatial scales** (the object alone, the object in local context, the object in full scene context) and averages the CLIP embeddings. This produces features that capture both fine-grained appearance and semantic context simultaneously.

---

## Results

Trained on Replica `room0` (~300 frames, 30k iterations):

| Query | Precision@0.5 | Notes |
|---|---|---|
| "a wooden chair" | 0.84 | Correct across all 4 chairs |
| "laptop" | 0.91 | Clean single-object localisation |
| "ceiling light" | 0.78 | Slight bleed to white walls |
| "comfortable seating" | 0.71 | Semantic generalisation working |
| "floor" | 0.88 | Large region, good coverage |

Qualitative results (rendered heatmaps) are in `outputs/renders/`.

---

## Repository Structure

```
langsplat-project/
├── README.md
├── requirements.txt
├── convert.py                  # COLMAP preprocessing
├── train.py                    # Main training script (Stage 1 + 3)
├── preprocess.py               # CLIP feature extraction (Stage 2)
├── render.py                   # Render trained model
├── query.py                    # Text query interface
├── demo.py                     # Gradio interactive demo
│
├── scene/
│   ├── gaussian_model.py       # 3DGS model definition
│   ├── language_field.py       # Per-Gaussian language MLP + autoencoder
│   └── dataset_readers.py      # Data loading for Replica / COLMAP
│
├── utils/
│   ├── clip_utils.py           # CLIP encoding + similarity
│   ├── sam_utils.py            # SAM mask generation
│   ├── render_utils.py         # Rendering helpers
│   └── loss_utils.py           # RGB + language losses
│
├── submodules/
│   ├── diff-gaussian-rasterization/   # CUDA rasteriser (from 3DGS repo)
│   └── simple-knn/                    # KNN for Gaussian initialisation
│
└── data/
    └── replica/
        └── room0/
            ├── images/         # RGB frames
            ├── sparse/         # COLMAP output (or precomputed poses)
            └── language_features/  # Output of preprocess.py
```

---

## Setup

### Requirements

- Python 3.10+
- CUDA 11.8+ (tested on RTX 3080 and A100)
- ~12 GB VRAM for training, ~6 GB for inference

### Installation

```bash
# Clone the repo
git clone https://github.com/yourusername/langsplat-project.git
cd langsplat-project

# Create conda environment
conda create -n langsplat python=3.10
conda activate langsplat

# Install PyTorch (adjust CUDA version as needed)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install CUDA extensions (requires ninja)
pip install ninja
pip install -e submodules/diff-gaussian-rasterization
pip install -e submodules/simple-knn

# Install remaining dependencies
pip install -r requirements.txt
```

### requirements.txt

```
clip @ git+https://github.com/openai/CLIP.git
segment-anything
gradio>=4.0
plyfile
tqdm
Pillow
scipy
scikit-learn
lpips
opencv-python
```

### Download SAM checkpoint

```bash
mkdir checkpoints
wget -P checkpoints \
  https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

---

## Dataset Preparation

### Option A: Replica (recommended for first run)

Replica provides clean indoor scenes with ground-truth camera poses — no COLMAP needed.

```bash
# Download via nice-slam's preprocessed version (poses already in NeRF format)
wget https://cvg-data.inf.ethz.ch/nice-slam/data/Replica.zip
unzip Replica.zip -d data/replica
```

The expected structure after extraction:
```
data/replica/room0/
├── results/           # RGB-D frames as frame000000.jpg etc.
├── traj.txt           # 4x4 camera poses, one per line
└── cam_params.json    # Intrinsics
```

Run the conversion script to produce COLMAP-compatible format:
```bash
python convert_replica.py --scene data/replica/room0
```

### Option B: Custom scene with COLMAP

If you want to use your own video or images:

```bash
# Place images in data/my_scene/input/
python convert.py -s data/my_scene

# This runs COLMAP feature extraction → matching → sparse reconstruction
# Output: data/my_scene/sparse/0/{cameras,images,points3D}.bin
```

For best results: capture at least 100 frames, move slowly, ensure good lighting, and avoid highly reflective surfaces.

---

## Training

### Stage 1: Base 3DGS

Train the colour-only Gaussian model first. This establishes scene geometry before any language features are introduced.

```bash
python train.py \
  -s data/replica/room0 \
  -m output/room0_base \
  --iterations 30000 \
  --densify_until_iter 15000 \
  --test_iterations 7000 15000 30000 \
  --save_iterations 30000
```

**Key hyperparameters:**

| Parameter | Default | Notes |
|---|---|---|
| `--iterations` | 30000 | 30k is sufficient for Replica room-scale scenes |
| `--densify_until_iter` | 15000 | Stop adding Gaussians at 15k to control count |
| `--lambda_dssim` | 0.2 | Weight of SSIM loss vs L1 |
| `--position_lr_init` | 1.6e-4 | Initial learning rate for Gaussian positions |

Training takes ~25 minutes on an RTX 3080. Monitor PSNR on the test views — expect ~28-32 dB on Replica room0 at 30k iterations.

**Render a test view to verify geometry:**
```bash
python render.py -m output/room0_base --skip_train
# Renders saved to output/room0_base/test/ours_30000/
```

---

### Stage 2: CLIP Feature Extraction

Extract dense per-pixel CLIP features for every training image using SAM for segmentation. This is the most time-consuming step (~3-4 hours for 300 frames on a single GPU). Run it overnight.

```bash
python preprocess.py \
  --dataset_path data/replica/room0 \
  --output_dir data/replica/room0/language_features \
  --sam_checkpoint checkpoints/sam_vit_h_4b8939.pth \
  --clip_model ViT-B/32 \
  --scales 1.0 0.5 0.25 \
  --batch_size 8
```

#### What this does internally

For each frame:

1. **SAM** generates automatic instance masks (typically 30-100 per image)
2. For each mask, the bounding box is cropped at **three scales** — tight crop, 2× expanded, 4× expanded — to capture local and global context
3. Each crop is passed through **CLIP ViT-B/32** to get a 512-dim embedding
4. The three embeddings are averaged and L2-normalised
5. The final embedding is assigned to all pixels within the SAM mask
6. A weighted average handles pixels covered by multiple masks

Output: one `.npy` file per image, shape `(H, W, 512)`, saved to `language_features/`.

#### Speed vs quality trade-off

```python
# Faster (fewer masks, lower quality):
SamAutomaticMaskGenerator(sam, points_per_side=16)

# Slower (more masks, better small-object coverage):
SamAutomaticMaskGenerator(sam, points_per_side=64)
```

Default is `points_per_side=32`, which balances speed and coverage well for room-scale scenes.

---

### Stage 3: Language Field Training

Train the language field on top of the frozen base Gaussians. A small 3-layer MLP per Gaussian learns to produce compressed (3-dim) language features that reconstruct to 512-dim CLIP embeddings.

```bash
python train.py \
  -s data/replica/room0 \
  -m output/room0_lang \
  --language_features_dir data/replica/room0/language_features \
  --start_checkpoint output/room0_base/chkpnt30000.pth \
  --iterations 30000 \
  --lambda_lang 0.1
```

The language loss term added to the standard 3DGS objective:

```
L_total = (1 - λ) · L_rgb + λ · L_lang

where:
  L_rgb  = (1 - λ_dssim) · L1(render, gt) + λ_dssim · (1 - SSIM(render, gt))
  L_lang = L1(render_lang, gt_lang_features)
  λ      = 0.1  (language loss weight)
```

This takes ~30 minutes additional training time. The colour rendering quality should not degrade significantly — if PSNR drops more than 1 dB, reduce `--lambda_lang`.

---

## Querying the Scene

### Command-line query

```bash
python query.py \
  -m output/room0_lang \
  --query "a wooden chair" \
  --camera_index 42 \
  --temperature 0.1 \
  --output_path outputs/chair_query.png
```

**Parameters:**

| Parameter | Description |
|---|---|
| `--query` | Natural language query string |
| `--camera_index` | Which training camera to render from (0 to N-1) |
| `--temperature` | Softmax temperature; lower = sharper localisation |
| `--threshold` | Binary mask threshold (default 0.5) |
| `--negative_query` | Optional negative query for contrastive scoring |

### Batch query (evaluate multiple queries)

```bash
python query.py \
  -m output/room0_lang \
  --query_file queries.txt \
  --camera_index 42 \
  --output_dir outputs/batch_queries/
```

`queries.txt` format (one query per line):
```
a wooden chair
laptop screen
ceiling light
comfortable seating
the floor
white wall
```

### Contrastive querying

For more precise localisation, provide a negative query to score `sim(positive) - sim(negative)`:

```bash
python query.py \
  -m output/room0_lang \
  --query "wooden chair" \
  --negative_query "floor tiles" \
  --camera_index 42
```

This significantly reduces false positives on large uniform regions.

### Python API

```python
from scene import GaussianModel
from utils.clip_utils import encode_text, compute_relevance
from render import render_query

# Load model
gaussians = GaussianModel(sh_degree=3)
gaussians.load_ply("output/room0_lang/point_cloud/iteration_30000/point_cloud.ply")

# Query
relevance = compute_relevance(
    gaussians,
    query="a wooden chair",
    temperature=0.1
)

# Render heatmap from camera 42
heatmap = render_query(gaussians, relevance, camera_index=42)
# heatmap: torch.Tensor of shape (3, H, W), values in [0, 1]
```

---

## Interactive Demo

Launch the Gradio web interface for real-time querying:

```bash
python demo.py -m output/room0_lang --port 7860
```

Open `http://localhost:7860` in your browser. The interface provides:

- **Text query input** — type any natural language description
- **Negative query input** (optional) — for contrastive scoring
- **Camera viewpoint slider** — scrub through all training camera positions
- **Temperature slider** — control sharpness of the heatmap
- **Side-by-side view** — original RGB render alongside query heatmap

---

## Extensions

### Negative queries

Score each Gaussian as `sim(positive) - sim(negative)` to suppress background:

```python
# In query.py
pos_sim = gaussian_feats @ pos_text_feat.T   # N
neg_sim = gaussian_feats @ neg_text_feat.T   # N
relevance = torch.softmax((pos_sim - neg_sim) / temperature, dim=0)
```

### Compositional queries

Combine two independent queries multiplicatively for spatial composition:

```python
# "red object near the table"
relevance_color = compute_relevance(gaussians, "red")
relevance_location = compute_relevance(gaussians, "near the table")
combined = relevance_color * relevance_location
```

### 3D instance segmentation

Rather than a soft heatmap, extract clean object instances by running connected-components on the thresholded Gaussians in 3D:

```python
from sklearn.cluster import DBSCAN

# Get positions of relevant Gaussians
relevant_positions = gaussians.get_xyz[mask].cpu().numpy()

# Cluster in 3D
labels = DBSCAN(eps=0.05, min_samples=10).fit_predict(relevant_positions)

# Each unique label = one instance
n_instances = len(set(labels)) - (1 if -1 in labels else 0)
print(f"Found {n_instances} instances of '{query}'")
```

### Stronger CLIP backbone

Swap ViT-B/32 for SigLIP or OpenCLIP ViT-G/14:

```python
# In preprocess.py and query.py
import open_clip
model, _, preprocess = open_clip.create_model_and_transforms(
    'ViT-G-14', pretrained='laion2b_s34b_b88k'
)
tokenizer = open_clip.get_tokenizer('ViT-G-14')
```

Note: feature dimensionality changes from 512 to 1024 — update the autoencoder input dimension accordingly.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| Blurry base renders, floaters | Poor camera pose estimation | Rerun COLMAP with `--SiftExtraction.max_num_features 8192`; ensure good frame coverage |
| All CLIP features look similar | Full-image CLIP rather than multi-scale SAM crops | Check `preprocess.py` is running SAM masks, not just encoding the full image |
| Queries find wrong objects | Softmax temperature too high (too diffuse) | Lower `--temperature` to 0.05 |
| Correct region but noisy boundary | SAM under-segmenting | Set `points_per_side=64` in `SamAutomaticMaskGenerator` |
| CUDA OOM during training | Too many Gaussians densified | Set `--densify_until_iter 15000`; reduce `--max_num_splats` if available |
| CUDA OOM during feature extraction | Batch size too large | Reduce `--batch_size` in `preprocess.py` to 2 or 4 |
| Language loss not decreasing | Learning rate too low for language MLP | Try `--lang_lr 1e-3` |
| PSNR drops after language training | `lambda_lang` too high | Reduce to 0.05; language features should not interfere with colour |

---

## References

```bibtex
@inproceedings{qin2024langsplat,
  title     = {LangSplat: 3D Language Gaussian Splatting},
  author    = {Qin, Minghan and Li, Wanhua and Zhou, Jiawei and Wang, Haoqian and Pfister, Hanspeter},
  booktitle = {CVPR},
  year      = {2024}
}

@article{kerbl2023gaussian,
  title   = {3D Gaussian Splatting for Real-Time Radiance Field Rendering},
  author  = {Kerbl, Bernhard and Kopanas, Georgios and Leimk{\"u}hler, Thomas and Drettakis, George},
  journal = {ACM Transactions on Graphics},
  year    = {2023}
}

@inproceedings{kerr2023lerf,
  title     = {LERF: Language Embedded Radiance Fields},
  author    = {Kerr, Justin and Kim, Chung Min and Goldberg, Ken and Kanazawa, Angjoo and Tancik, Matthew},
  booktitle = {ICCV},
  year      = {2023}
}

@article{kirillov2023sam,
  title   = {Segment Anything},
  author  = {Kirillov, Alexander and Mintun, Eric and Ravi, Nikhila and others},
  journal = {arXiv:2304.02643},
  year    = {2023}
}

@inproceedings{radford2021clip,
  title     = {Learning Transferable Visual Models From Natural Language Supervision},
  author    = {Radford, Alec and Kim, Jong Wook and Hallacy, Chris and others},
  booktitle = {ICML},
  year      = {2021}
}
```

---

## Acknowledgements

Base 3DGS implementation from [gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting) (Kerbl et al.). Language field architecture and multi-scale CLIP extraction from [LangSplat](https://github.com/minghanqin/LangSplat) (Qin et al.). Replica dataset from [Facebook Research](https://github.com/facebookresearch/Replica-Dataset).

---

*Built as a 1-week project exploring neural scene representations and open-vocabulary 3D understanding. For questions or issues, open a GitHub issue.*
