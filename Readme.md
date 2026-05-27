# WAF-X-Net with Attention-Aware Bayesian Optimization

This repository contains the implementation and reproduction materials for:

**WAF-X-Net: Weighted Attention Fusion Xception Model with Attention-Aware Bayesian Optimization for Frame-Level Deepfake Detection**

WAF-X-Net is a frame-level deepfake detection framework that combines:

- a pretrained Xception backbone,
- multi-head self-attention,
- Discriminatively Weighted Attention Fusion (DWAF),
- and Attention-Aware Bayesian Optimization (AABO).

The framework is designed to address challenges in deepfake detection arising from subtle and spatially localized manipulation artifacts, uniform attention-head aggregation, and cross-dataset domain shift.

---

## Key Components

### 1. Xception Backbone

A pretrained Xception network is used to extract spatial facial representations from sampled video frames.

### 2. Multi-Head Self-Attention

Multi-head self-attention is applied to model long-range contextual dependencies across spatial regions.

### 3. Discriminatively Weighted Attention Fusion

Unlike uniform averaging or representation-preserving concatenation, DWAF assigns adaptive fusion weights to attention heads based on class-dependent discriminative behaviour between real and fake samples. This allows heads with stronger real-fake discriminability to contribute more strongly during feature aggregation.

### 4. Attention-Aware Bayesian Optimization

AABO extends conventional Bayesian Optimization by jointly optimizing:

- validation AUC,
- and attention discriminability.

The optimization process is implemented using Optuna with its default TPE sampler configuration.

The AABO objective is:

```text
J(theta) = 0.7 × validation_AUC + 0.3 × attention_discriminability
