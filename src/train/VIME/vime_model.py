"""
vime_model.py
PyTorch architecture for VIME (Value Imputation and Mask Estimation).

References:
    Yoon, J., Zhang, Y., Jordon, J., & van der Schaar, M. (2020).
    VIME: Extending the Success of Self- and Semi-supervised Learning to Tabular Domain.
    NeurIPS 2020.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple


class VIMEEncoder(nn.Module):
    """
    Shared tabular representation encoder e(x).
    Transforms input features into a latent representation.
    """
    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super(VIMEEncoder, self).__init__()
        layers = []
        in_d = input_dim
        
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(in_d, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            in_d = hidden_dim
            
        layers.append(nn.Linear(in_d, latent_dim))
        layers.append(nn.BatchNorm1d(latent_dim))
        layers.append(nn.GELU())
        
        self.encoder_net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder_net(x)


class VIMEMaskEstimator(nn.Module):
    """
    Mask Estimation Head f_m(h).
    Estimates the binary mask vector m in [0, 1]^d indicating which features were corrupted.
    """
    def __init__(self, latent_dim: int, input_dim: int, hidden_dim: int = 128):
        super(VIMEMaskEstimator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()  # Probability that feature j was corrupted
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h)


class VIMEValueImputer(nn.Module):
    """
    Value / Feature Imputation Head f_r(h).
    Reconstructs the original uncorrupted feature vector x in R^d.
    """
    def __init__(self, latent_dim: int, input_dim: int, hidden_dim: int = 128):
        super(VIMEValueImputer, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h)


class VIMESelfSupervised(nn.Module):
    """
    Full Self-Supervised VIME Model.
    Combines the shared encoder, mask estimator head, and value imputer head.
    """
    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super(VIMESelfSupervised, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        self.encoder = VIMEEncoder(
            input_dim=input_dim,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.mask_estimator = VIMEMaskEstimator(
            latent_dim=latent_dim,
            input_dim=input_dim,
            hidden_dim=hidden_dim // 2,
        )
        self.value_imputer = VIMEValueImputer(
            latent_dim=latent_dim,
            input_dim=input_dim,
            hidden_dim=hidden_dim // 2,
        )

    def forward(self, x_corrupted: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for pretext training.
        Returns:
            mask_pred (torch.Tensor): Predicted mask probabilities [B, D]
            value_pred (torch.Tensor): Reconstructed feature values [B, D]
        """
        h = self.encoder(x_corrupted)
        mask_pred = self.mask_estimator(h)
        value_pred = self.value_imputer(h)
        return mask_pred, value_pred


class VIMEPredictor(nn.Module):
    """
    Downstream Predictive Model using the VIME Pretrained Encoder.
    Fine-tunes or linear probes for tabular regression / classification.
    """
    def __init__(
        self,
        encoder: VIMEEncoder,
        output_dim: int,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        freeze_encoder: bool = False,
    ):
        super(VIMEPredictor, self).__init__()
        self.encoder = encoder
        
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        self.head = nn.Sequential(
            nn.Linear(self.encoder.encoder_net[-2].out_features if hasattr(self.encoder.encoder_net[-2], 'out_features') else 128, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x)
        out = self.head(h)
        return out
