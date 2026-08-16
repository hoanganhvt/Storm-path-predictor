# VIME (Value Imputation and Mask Estimation) Fine-Tuning for Storm Path Prediction

This directory implements the **VIME** tabular self-supervised and semi-supervised deep learning framework for multi-horizon storm trajectory displacement prediction ($\Delta \text{lat}, \Delta \text{lon}$ across 6h, 12h, 18h, and 24h horizons).

---

## 1. Overview of Architecture & Pipeline

### Phase 1: Self-Supervised Pretraining (`train_vime_pretrain.py`)
- Learns representations on the storm feature manifold using two pretext tasks:
  1. **Mask Estimation Task ($\mathcal{L}_m$)**: Predicts binary corruption mask $m \sim \text{Bernoulli}(p_m)$ via BCE Loss.
  2. **Feature Value Imputation ($\mathcal{L}_r$)**: Reconstructs original uncorrupted feature values $x$ from $\tilde{x} = x \odot (1-m) + \bar{x} \odot m$ via MSE Loss, where $\bar{x}$ is sampled from the empirical marginal distributions.
- Pretraining Objective:
  $$\mathcal{L}_{\text{pretrain}} = \mathcal{L}_m(m, \hat{m}) + \alpha \cdot \mathcal{L}_r(x, \hat{x})$$

### Phase 2: Semi-Supervised Fine-Tuning on Test Data (`vime_finetune.py`)
- Attaches a multi-horizon predictor head $f(e(x))$ onto the encoder $e(x)$ to predict 8 target coordinates:
  `['delta_lat_6h', 'delta_lon_6h', 'delta_lat_12h', 'delta_lon_12h', 'delta_lat_18h', 'delta_lon_18h', 'delta_lat_24h', 'delta_lon_24h']`
- **Supervised Task Loss**:
  $$\mathcal{L}_s = \text{MSE}(\hat{y}_{\text{train}}, y_{\text{train}})$$
- **Consistency Regularization on Unlabeled Test Data**:
  For each test sample $x_{\text{test}}$, generates $K$ perturbed variants $x_{\text{test}}^{(k)}$ via empirical marginal corruption and enforces output consistency:
  $$\mathcal{L}_c = \frac{1}{K} \sum_{k=1}^K \| f(e(x_{\text{test}})) - f(e(x_{\text{test}}^{(k)})) \|^2$$
- **Test Domain Reconstruction Adaptation**:
  $$\mathcal{L}_{\text{recon, test}} = \text{MSE}(\hat{x}_{\text{test}}, x_{\text{test}})$$
- **Total Fine-Tuning Objective**:
  $$\mathcal{L}_{\text{total}} = \mathcal{L}_s + \beta \mathcal{L}_c + \gamma \mathcal{L}_{\text{recon, test}}$$

---

## 2. File Structure

```
src/train/VIME finetune/
├── vime_model.py             # VIMEEncoder, VIMEMaskEstimator, VIMEFeatureReconstructor, VIMEPredictor, VIMEModel
├── vime_utils.py             # Data loading, cyclical & lag feature engineering, VIMECorruptor, Haversine/Cosine evaluation
├── train_vime_pretrain.py    # Self-supervised pretraining script
├── vime_finetune.py          # Main semi-supervised fine-tuning on test data script
├── test_vime.py              # Verification suite (syntax, tensor shapes, forward/backward, corruptions)
├── README.md                 # Documentation
└── checkpoints/              # Checkpoint directory (best_vime_finetuned.pth, vime_pretrained.pth, scaler.pkl)
```

---

## 3. How to Run

### Step 1 (Optional): Self-Supervised Pretraining
To pretrain the VIME encoder on the feature distributions:
```bash
python "src/train/VIME finetune/train_vime_pretrain.py" --epochs 50 --batch_size 64 --lr 0.001 --p_m 0.3 --alpha 2.0
```

### Step 2: Fine-Tuning VIME on Test Data
To fine-tune VIME using supervised storm tracks and test data consistency regularization:
```bash
python "src/train/VIME finetune/vime_finetune.py" --epochs 50 --batch_size 64 --lr 0.0005 --beta 1.0 --gamma 0.2 --K 3
```

### Key Arguments:
- `--epochs`: Number of fine-tuning epochs (default: `50`).
- `--batch_size`: Batch size (default: `64`).
- `--lr`: Learning rate (default: `5e-4`).
- `--beta`: Weight for test consistency loss (default: `1.0`).
- `--gamma`: Weight for test domain reconstruction loss (default: `0.2`).
- `--K`: Number of perturbed test samples generated per batch (default: `3`).
- `--p_m`: Feature corruption probability (default: `0.3`).
- `--no_cuda`: Force CPU mode.
