# WAF-X-Net with AABO

WAF-X-Net is a deepfake detection framework that combines a pretrained Xception backbone with multi-head self-attention, Discriminatively Weighted Attention Fusion (DWAF), and Attention-Aware Bayesian Optimization (AABO) for manipulation-sensitive feature learning.

The framework is designed to address challenges in deepfake detection arising from:
- subtle and spatially localized manipulation artifacts,
- uniform attention aggregation,
- and cross-dataset domain shift.

---

# Key Components

## 1. Xception Backbone
A pretrained Xception network is used to extract spatial facial representations from sampled video frames.

## 2. Multi-Head Self-Attention (MHA)
Multi-head self-attention is applied to model long-range contextual dependencies across spatial regions.

## 3. Discriminatively Weighted Attention Fusion (DWAF)
Unlike standard uniform attention aggregation, DWAF assigns adaptive weights to attention heads based on class-dependent discriminative behaviour between real and fake samples.

This enables:
- selective emphasis on manipulation-sensitive attention responses,
- improved feature aggregation,
- and stronger discriminative representation learning.

## 4. Attention-Aware Bayesian Optimization (AABO)
AABO extends conventional Bayesian Optimization by jointly optimizing:
- validation AUC,
- and attention discriminability.

The optimization process is implemented using Optuna with the default TPE sampler configuration.

---

# Experimental Pipeline

The repository contains scripts for:
1. Hyperparameter optimization (BOAUC and AABO)
2. Full 35-epoch retraining
3. Held-out Celeb-DF v2 evaluation
4. Cross-dataset evaluation on FaceForensics++ Face2Face
5. Component ablation experiments

The experimental workflow follows:

Optimization → Retraining → Held-out Testing → Cross-Dataset Evaluation → Ablation Analysis

---

# Datasets

## Celeb-DF v2
Used for:
- optimization,
- full retraining,
- and held-out evaluation.

Experimental protocol:
- 500 real + 500 fake videos used for training/validation
- fixed 80:20 video-level split
- 30 sampled frames per video

Held-out evaluation:
- 100 real + 100 fake videos
- 30 sampled frames per video
- evaluated separately from optimization and validation stages

## FaceForensics++ (Face2Face subset)
Used for:
- zero-shot cross-dataset evaluation,
- and limited fine-tuning experiments.

Experimental subset:
- 500 real + 500 fake videos
- 50 real + 50 fake videos used for fine-tuning
- 450 real + 450 fake videos used for evaluation

---

# Reproducibility

Experiments are conducted using:
- video-level train/validation/test partitioning
- random seeds: {42, 123}
- Optuna TPE sampler for Bayesian Optimization
- 15 optimization trials
- 35-epoch final retraining
- held-out test evaluation
- mean ± standard deviation reporting across two seeds

---

# Repository Structure

```text
optimization/
    BOAUC optimization scripts
    AABO optimization scripts

training/
    Final 35-epoch retraining scripts

evaluation/
    Held-out Celeb-DF v2 evaluation
    Cross-dataset FaceForensics++ evaluation

ablation/
    Component ablation experiments

models/
    Saved model checkpoints

results/
    Validation results
    Held-out evaluation results
    Cross-dataset results
    Ablation results
```

---

# Installation

```bash
git clone https://github.com/jyothibiyyani/unsup_deepfake.git
cd unsup_deepfake
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Training

## BOAUC Optimization

```bash
python optimization/boauc_optimization.py
```

## AABO Optimization

```bash
python optimization/aabo_optimization.py
```

## Final Retraining

```bash
python training/final_retraining.py
```

---

# Evaluation

## Held-out Celeb-DF v2 Evaluation

```bash
python evaluation/heldout_test_evaluation.py
```

## Cross-Dataset Evaluation

```bash
python evaluation/cross_dataset_evaluation.py
```

---

# Ablation Studies

Run component ablation experiments:

```bash
python ablation/ablation_experiments.py
```

The ablation study evaluates:
- Xception baseline
- Xception + MHA
- WAF-X-Net
- WAF-X-Net + BOAUC
- WAF-X-Net + AABO

---

# Main Features

- Discriminability-guided attention fusion
- Attention-aware hyperparameter optimization
- Multi-head self-attention
- Multi-seed evaluation
- Held-out test evaluation
- Cross-dataset analysis
- Weight entropy analysis
- Attention discriminability analysis

---

# Code Availability

The implementation code, including:
- training,
- optimization,
- held-out evaluation,
- cross-dataset evaluation,
- and ablation scripts,

is publicly available in this repository.

---

# Citation

If you use this repository, please cite:

```bibtex
@article{wafxnet2025,
  title={WAF-X-Net: Weighted Attention Fusion Xception Model with Attention-Aware Bayesian Optimization for Robust Deepfake Detection},
  author={Jyothi, B. N. and Jabbar, M. A.},
  journal={International Journal of Intelligent Engineering and Systems},
  year={2025}
}
```

---

# License

This repository is released for academic and research purposes.
