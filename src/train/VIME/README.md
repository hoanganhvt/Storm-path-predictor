# VIME: Value Imputation and Mask Estimation for Tabular Storm Data

This directory contains the complete PyTorch implementation of the **VIME (Value Imputation and Mask Estimation)** framework for tabular data, as introduced in *Yoon et al. (NeurIPS 2020)*.

---

## 📁 Directory Structure

- [`vime_model.py`](file:///c:/Users/ADMIN/Documents/Storm%20path%20predictor/src/train/VIME/vime_model.py): Model architectures including `VIMEEncoder`, `VIMEMaskEstimator`, `VIMEValueImputer`, `VIMESelfSupervised`, and `VIMEPredictor`.
- [`vime_dataset.py`](file:///c:/Users/ADMIN/Documents/Storm%20path%20predictor/src/train/VIME/vime_dataset.py): Empirical marginal corruption generator, PyTorch Datasets, and composite VIME loss functions.
- [`train_vime.py`](file:///c:/Users/ADMIN/Documents/Storm%20path%20predictor/src/train/VIME/train_vime.py): End-to-end self-supervised pretext training, downstream fine-tuning with consistency regularization, and test evaluation.
- `checkpoints/`: Directory storing trained model checkpoints and scaler artifacts (`best_vime_pretext.pth`, `best_vime_predictor.pth`, `vime_scaler.pkl`).
- `train_result.txt`: Output evaluation report showing cosine similarities and Haversine distance errors (in km) across forecast horizons (6h, 12h, 18h, 24h).

---

## 🚀 How to Train

Run the training pipeline with default parameters on `Datasets/Storm_data/train.csv`:

```bash
python src/train/VIME/train_vime.py
```

### Customizable Hyperparameters

```bash
python src/train/VIME/train_vime.py \
  --epochs_pretext 40 \
  --epochs_downstream 50 \
  --batch_size 64 \
  --p_m 0.3 \
  --alpha 2.0 \
  --beta 0.5 \
  --latent_dim 128 \
  --hidden_dim 256 \
  --lr_pretext 0.001 \
  --lr_downstream 0.0005
```

---

## 🔬 Mathematical Formulation

### 1. Empirical Marginal Corruption
For an input vector $\mathbf{x} \in \mathbb{R}^d$:
1. Sample binary mask $\mathbf{m} \sim \text{Bernoulli}(p_m)^d$.
2. For each corrupted feature ($m_j = 1$), sample $\bar{x}_j$ from the empirical marginal distribution of feature $j$ across the dataset.
3. Compute corrupted input $\mathbf{\tilde{x}} = \mathbf{m} \odot \mathbf{\bar{x}} + (1 - \mathbf{m}) \odot \mathbf{x}$.

### 2. Pretext Loss
$$\mathcal{L}_{\text{self}} = \mathcal{L}_{\text{mask}}(\mathbf{m}, \mathbf{\hat{m}}) + \alpha \cdot \mathcal{L}_{\text{impute}}(\mathbf{x}, \mathbf{\hat{x}})$$

### 3. Downstream Consistency Regularization
$$\mathcal{L}_{\text{downstream}} = \mathcal{L}_{\text{sup}}(\mathbf{y}, \mathbf{\hat{y}}) + \beta \cdot \frac{1}{K} \sum_{k=1}^K \left\| f_y(e(\mathbf{\tilde{x}}^{(k)})) - f_y(e(\mathbf{x})) \right\|_2^2$$
