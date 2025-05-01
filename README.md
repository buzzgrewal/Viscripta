# Viscripta: Where vision meets vivid narration — intelligent image captions reimagined.
---

# 🧠 Fine-Tuning BLIP-2 with Flan-T5 on Flickr8k Dataset

A comprehensive implementation of a **Vision-Language Model (VLM)** fine-tuning pipeline using **BLIP-2 (with Flan-T5-XL)** on the **Flickr8k dataset**, optimized for image captioning tasks. This repository is part of an academic assignment focused on generating detailed image descriptions and visual narratives through multimodal learning.

---

## 📌 Project Overview

This project fine-tunes **BLIP-2**, a state-of-the-art generative VLM, to produce detailed and coherent image captions using the Flickr8k dataset. Key goals:

- Generate **rich image descriptions** using multimodal input (image + text)
- Explore **fine-tuning and decoding strategies**
- Evaluate outputs using **automatic metrics and manual analysis**
- Optimize training pipeline for **Google Colab** (low VRAM setups)

---

## 🖼 Dataset: Flickr8k

Flickr8k contains:
- 8,092 images
- 5 human-annotated captions per image
- Used subset of **8,000 (image, caption)** pairs

📎 [Flickr8k Dataset on Kaggle](https://www.kaggle.com/datasets/adityajn105/flickr8k)

---

## 🧰 Tools & Frameworks

| Tool | Purpose |
|------|---------|
| PyTorch | Model training & data handling |
| HuggingFace Transformers | Tokenization & decoding |
| Salesforce LAVIS | Loading & using BLIP-2 |
| Torchvision | Image processing |
| Evaluate (HF) | BLEU, METEOR, ROUGE-L scoring |
| Google Colab | GPU-backed training |

---

## ⚙️ Setup Instructions

### 🖥 Environment Setup

Install dependencies:

```bash
pip install torch torchvision transformers datasets evaluate nltk sacrebleu rouge-score git+https://github.com/salesforce/LAVIS.git
```

Download NLTK resources:

```python
import nltk
nltk.download('punkt')
```

### 📁 Directory Structure

```
.
├── data/                    # Contains train.csv, test.csv, and Flickr8k images
├── models/                  # Checkpoints and saved weights
├── notebooks/               # Jupyter notebook for training and experiments
├── scripts/                 # Modular Python scripts
│   ├── data_module.py
│   ├── train.py
│   ├── generate.py
│   ├── evaluate.py
├── utils/                   # Decoding and evaluation utilities
├── requirements.txt
└── README.md
```

---

## 🧠 Model: BLIP-2 + Flan-T5-XL

We used `blip2_t5` from the **LAVIS** framework:

```python
model, vis_processors, txt_processors = load_model_and_preprocess(
    name="blip2_t5", model_type="flan_t5_xl", is_eval=False, device=device
)
```

> BLIP-2 decouples vision encoding and language generation, allowing use of large language models like Flan-T5.

---

## 📋 Fine-Tuning Procedure

1. **Image preprocessing** using LAVIS visual processors.
2. **Text tokenization** using Flan-T5 tokenizer.
3. **Training loop** with `torch.cuda.amp` for mixed precision.
4. **Loss minimization** with AdamW optimizer.

We used a conservative learning rate (`1e-5`) and small batch sizes to work within Google Colab constraints.

---

## 🌀 Decoding Strategies

Implemented multiple decoding options:

| Method     | Description |
|------------|-------------|
| Beam Search | More deterministic, less diverse |
| Top-k Sampling | Keeps top-k tokens by likelihood |
| Top-p (Nucleus) | Samples from top cumulative probability |
| Temperature | Controls randomness in sampling |

Example:

```python
model.generate(
    {"image": image_tensor},
    use_nucleus_sampling=True,
    top_k=50,
    temperature=0.8
)
```

---

## 📊 Evaluation Metrics

We used both automated and manual evaluation:

### ✅ Automatic Metrics

| Metric | Description |
|--------|-------------|
| BLEU-4 | N-gram overlap score |
| METEOR | Weighted F1 score with synonym matching |
| ROUGE-L | Longest common subsequence |
| SPICE | Semantic propositional content |
| CIDEr | Consensus-based evaluation |
| Self-BLEU | Measures diversity of generations |
| Distinct-n | Uniqueness in generated words |

Sample usage:

```python
from evaluate import load
meteor = load("meteor")
meteor.compute(predictions=preds, references=refs)
```

---

## 🔍 Qualitative Analysis

Analyzed 20 random test images for:

- ✅ Mention of ≥3 distinct elements
- ❌ Hallucinations (objects not in image)
- ♻️ Repetition or bland output
- ❓ Omission of key visual cues

Visualized GT vs. generated captions using `matplotlib`.

---

## 📈 Results

| Metric | Value |
|--------|-------|
| BLEU-4 | 27.5  |
| METEOR | 31.2  |
| ROUGE-L | 45.7 |
| SPICE | 20.9  |
| CIDEr | 65.1  |

✅ Met assignment goal of BLEU > 25 and CIDEr > 60

---

## 📦 Modular Scripts

| Script | Purpose |
|--------|---------|
| `data_module.py` | Dataset loading and split |
| `train.py` | Fine-tuning loop |
| `generate.py` | Caption generation |
| `evaluate.py` | Metric computation |
| `utils.py` | Decoding, formatting |

All scripts are reproducible with random seeds and support CLI flags for hyperparameter tuning.

---

## ✅ Success Criteria Achieved

- [x] Rich, creative captions (≥ 3 distinct elements)
- [x] Low hallucination and repetition rate
- [x] BLEU-4 > 25, CIDEr > 60
- [x] Modular, reproducible code

---

## 🧠 Future Improvements

- 🔁 Integrate **LoRA** for lightweight fine-tuning
- 🧪 Run human evaluation for quality control
- 🌍 Add **multilingual captioning**
- 🔊 Integrate with **TTS and ASR** for multimodal interaction

---

## 🔗 Useful Links

- 📘 Flickr8k: [https://www.kaggle.com/datasets/adityajn105/flickr8k](https://www.kaggle.com/datasets/adityajn105/flickr8k)
- 🧠 LAVIS (BLIP-2): [https://github.com/salesforce/LAVIS](https://github.com/salesforce/LAVIS)
- 📝 Meddium: https://buzzgrewal.medium.com/fine-tuning-blip-2-on-flickr8k-teaching-vision-language-models-to-describe-the-world-fc3e42350970

---

## 📄 License

This project is for academic/research use only. Please cite BLIP-2 and Flickr8k if using this codebase.

---

## 👋 Acknowledgements

- Salesforce Research for BLIP-2
- HuggingFace for the Transformers and Evaluate libraries
- Kaggle dataset authors for Flickr8k
- Instructor for a challenging and rewarding assignment!

---
