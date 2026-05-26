# WAF-X-Net with Attention-Aware Bayesian Optimization

This repository contains the implementation files for:

**WAF-X-Net: Weighted Attention Fusion Xception Model with Attention-Aware Bayesian Optimization for Frame-Level Deepfake Detection**

WAF-X-Net is a frame-level deepfake detection framework that combines:

- a pretrained Xception backbone,
- multi-head self-attention,
- Discriminatively Weighted Attention Fusion (DWAF),
- and Attention-Aware Bayesian Optimization (AABO).

The framework is designed to address challenges in deepfake detection caused by subtle localized manipulation artifacts, uniform attention-head aggregation, and cross-dataset domain shift.

---

## Key Components

### 1. Xception Backbone

A pretrained Xception network is used to extract spatial facial representations from sampled video frames.

### 2. Multi-Head Self-Attention

Multi-head self-attention is applied to model long-range spatial dependencies across facial regions.

### 3. Discriminatively Weighted Attention Fusion

DWAF assigns adaptive fusion weights to attention heads based on class-dependent discriminative behaviour between real and fake samples. This allows attention heads with stronger real-fake discriminability to contribute more strongly during feature aggregation.

### 4. Attention-Aware Bayesian Optimization

AABO extends conventional Bayesian Optimization by jointly optimizing:

- validation AUC,
- and attention discriminability.

The optimization process is implemented using Optuna with the default TPE sampler configuration.

---

## Repository Structure

```text
WAF-X-Net-AABO/
│
├── configs/                         # Selected hyperparameters and experiment settings
├── docs/                            # Dataset preparation and reproduction documentation
├── models/                          # WAF-X-Net model definition
├── Optimization/                    # BOAUC and AABO hyperparameter search
├── Full_Model_Training/             # Final 35-epoch retraining
├── heldout_testset_evaluation/      # Held-out Celeb-DF v2 evaluation
├── cross_dataset_evaluation/        # FaceForensics++ Face2Face evaluation
├── ablation/                        # Component ablation experiments
├── splits/                          # Video-level split files and subset identifiers
├── results/                         # Table-level and seed-wise result files
└── tools/                           # Utility scripts
