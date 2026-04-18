<div align="center">

# CrossFlowDG: Bridging the Modality Gap with Cross-modal Flow Matching for Domain Generalization



[Antonios Kritikos](https://scholar.google.com/citations?user=Ra0Zwb0AAAAJ&hl=en), [Nikolaos Spanos](https://scholar.google.com/citations?user=QDPr9W8AAAAJ&hl=en), [Athanasios Voulodimos](https://scholar.google.com/citations?hl=el&user=R97D37EAAAAJ)<br />
🏫 [ECE, NTUA](http://ece.ntua.gr/)

</div>

**Abstract**: Domain generalization (DG) aims to maintain performance under domain shift, which in computer vision appears primarily as stylistic variations that cause models to overfit to domain-specific appearance cues rather than class semantics. To overcome this, recent methods use textual representations as stable, domain-invariant anchors. However, multimodal approaches that rely on cosine similarity-based contrastive alignment leave a modality gap where image and text embeddings remain geometrically separated despite semantic correspondence. We propose CrossFlowDG, a novel DG framework that addresses this residual gap using noise-free, cross-modal flow matching. By learning a continuous transformation in the joint Euclidean latent space, our framework explicitly transports domain-biased image embeddings toward domain-invariant text embeddings of the correct class. Using the efficient VMamba image encoder and CLIP's text encoder, CrossFlowDG is tested against four common DG benchmarks, and achieves competitive performance on several benchmarks and state-of-the-art on TerraIncognita.

# ⚙️ Installation
```bash
# Clone the repository
git clone https://github.com/ajkrit/CrossFlowDG.git
cd CrossFlowDG

# (Recommended) Create and activate a virtual environment
conda create -n virtual_env python=3.12 -y
conda activate virtual_env

# Install requirements
pip install -r requirements.txt
```

# 🗂️ Dataset

# 🧠 Training

# 📊 Evaluation

# 📄 Citation

If you find this work useful for your research, please consider citing our paper:
```
```

This code has been adapted from [here](https://github.com/mzeromiko/vmamba) and [here](https://github.com/longshaocong/DGMamba).
