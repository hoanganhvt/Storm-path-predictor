"""
vime_dataset.py
Data processing, empirical marginal corruption mechanism, and PyTorch Datasets for VIME.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from typing import Tuple, Optional


def corrupt_tabular_data(
    data: np.ndarray,
    p_m: float = 0.3,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Corrupts tabular data according to the VIME empirical marginal distribution sampling.

    For each sample x_i in R^d:
    1. Draw binary mask m_i ~ Bernoulli(p_m)^d.
    2. For each feature j where m_ij = 1, sample x_bar_ij from the empirical
       marginal distribution of feature j (i.e. pick feature j from a random row k).
    3. Return corrupted vector x_tilde and mask m.

    Parameters
    ----------
    data : np.ndarray
        Original data matrix of shape [N, D].
    p_m : float
        Corruption probability (default 0.3).
    rng : np.random.Generator, optional
        Random generator instance.

    Returns
    -------
    x_tilde : np.ndarray
        Corrupted data matrix of shape [N, D].
    mask : np.ndarray
        Binary mask matrix of shape [N, D] (1 if corrupted, 0 if original).
    """
    if rng is None:
        rng = np.random.default_rng()

    n_samples, n_features = data.shape
    
    # 1. Sample binary mask vector m ~ Bernoulli(p_m)
    mask = rng.binomial(n=1, p=p_m, size=(n_samples, n_features)).astype(np.float32)

    # 2. Sample corrupted values from empirical marginal distributions
    # For each feature j, randomly permute row indices
    x_bar = np.zeros_like(data)
    for j in range(n_features):
        random_indices = rng.integers(low=0, high=n_samples, size=n_samples)
        x_bar[:, j] = data[random_indices, j]

    # 3. Create corrupted data: x_tilde = m * x_bar + (1 - m) * x
    x_tilde = mask * x_bar + (1.0 - mask) * data

    return x_tilde.astype(np.float32), mask.astype(np.float32)


class VIMESelfSupervisedDataset(Dataset):
    """
    PyTorch Dataset for Self-Supervised VIME Pretraining.
    Supports on-the-fly or static marginal corruption generation.
    """
    def __init__(
        self,
        data: np.ndarray,
        p_m: float = 0.3,
        on_the_fly_corruption: bool = True,
        seed: Optional[int] = None,
    ):
        self.data = np.asarray(data, dtype=np.float32)
        self.n_samples, self.n_features = self.data.shape
        self.p_m = p_m
        self.on_the_fly = on_the_fly_corruption
        self.rng = np.random.default_rng(seed)

        if not self.on_the_fly:
            self.x_tilde, self.mask = corrupt_tabular_data(self.data, p_m=self.p_m, rng=self.rng)

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x_orig = self.data[idx]

        if self.on_the_fly:
            # Generate corruption per sample against dataset empirical distribution
            mask = self.rng.binomial(n=1, p=self.p_m, size=self.n_features).astype(np.float32)
            rand_indices = self.rng.integers(0, self.n_samples, size=self.n_features)
            x_bar = self.data[rand_indices, np.arange(self.n_features)]
            x_corrupted = mask * x_bar + (1.0 - mask) * x_orig
            return (
                torch.from_numpy(x_orig),
                torch.from_numpy(x_corrupted),
                torch.from_numpy(mask),
            )
        else:
            return (
                torch.from_numpy(x_orig),
                torch.from_numpy(self.x_tilde[idx]),
                torch.from_numpy(self.mask[idx]),
            )


class VIMESemiSupervisedDataset(Dataset):
    """
    PyTorch Dataset for Downstream Supervised & Semi-Supervised fine-tuning.
    """
    def __init__(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        is_labeled: Optional[np.ndarray] = None,
    ):
        self.X = np.asarray(X, dtype=np.float32)
        self.y = np.asarray(y, dtype=np.float32) if y is not None else None
        self.is_labeled = np.asarray(is_labeled, dtype=bool) if is_labeled is not None else np.ones(len(self.X), dtype=bool)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        x_val = torch.from_numpy(self.X[idx])
        if self.y is not None:
            y_val = torch.from_numpy(self.y[idx])
            lbl_val = torch.tensor(self.is_labeled[idx], dtype=torch.bool)
            return x_val, y_val, lbl_val
        return x_val


class VIMELoss(nn.Module):
    """
    Combined Self-Supervised VIME Loss:
    L_self = L_mask + alpha * L_impute

    Handles missing values (encoded e.g. as -1 or NaN) if specified.
    """
    def __init__(self, alpha: float = 2.0, missing_val_threshold: Optional[float] = None):
        super(VIMELoss, self).__init__()
        self.alpha = alpha
        self.missing_val_threshold = missing_val_threshold
        self.bce_loss = nn.BCELoss(reduction='none')
        self.mse_loss = nn.MSELoss(reduction='none')

    def forward(
        self,
        mask_pred: torch.Tensor,
        value_pred: torch.Tensor,
        mask_true: torch.Tensor,
        value_true: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # 1. Mask Estimation Loss (Binary Cross-Entropy)
        # Clamp mask_pred for numerical stability in BCE
        mask_pred_clamped = torch.clamp(mask_pred, 1e-7, 1.0 - 1e-7)
        loss_mask_raw = self.bce_loss(mask_pred_clamped, mask_true)
        loss_mask = torch.mean(loss_mask_raw)

        # 2. Value Imputation Loss (MSE on reconstructed values)
        loss_impute_raw = self.mse_loss(value_pred, value_true)

        if self.missing_val_threshold is not None:
            # If missing values are marked below a threshold (e.g., -1.0 for missing)
            valid_mask = (value_true >= self.missing_val_threshold).float()
            loss_impute = (loss_impute_raw * valid_mask).sum() / (valid_mask.sum() + 1e-8)
        else:
            loss_impute = torch.mean(loss_impute_raw)

        # Total Loss
        total_loss = loss_mask + self.alpha * loss_impute
        return total_loss, loss_mask, loss_impute
