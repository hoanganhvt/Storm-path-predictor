import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict


class VIMEEncoder(nn.Module):
    """
    VIME Shared Encoder:
    Maps input feature vector of dimension `input_dim` to a latent representation `latent_dim`.
    Uses Linear layers, LayerNorm, GELU activation, and Dropout for robust feature representation.
    """
    def __init__(self, input_dim: int, latent_dim: int = 128, hidden_dim: int = 256, dropout: float = 0.1):
        super(VIMEEncoder, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class VIMEMaskEstimator(nn.Module):
    """
    VIME Mask Estimator (s_m):
    Predicts the binary corruption mask m from the latent representation z.
    Outputs values in [0, 1] using Sigmoid activation.
    """
    def __init__(self, latent_dim: int = 128, hidden_dim: int = 128, output_dim: int = 40):
        super(VIMEMaskEstimator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid()
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class VIMEFeatureReconstructor(nn.Module):
    """
    VIME Feature Reconstructor / Value Imputer (s_r):
    Reconstructs the original uncorrupted feature vector x from the latent representation z.
    """
    def __init__(self, latent_dim: int = 128, hidden_dim: int = 128, output_dim: int = 40):
        super(VIMEFeatureReconstructor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class VIMEPredictor(nn.Module):
    """
    VIME Downstream Predictor Head (f):
    Predicts the multi-horizon storm displacements (delta_lat, delta_lon for 6h, 12h, 18h, 24h).
    """
    def __init__(self, latent_dim: int = 128, hidden_dim: int = 128, output_dim: int = 8, dropout: float = 0.1):
        super(VIMEPredictor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, output_dim)
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class VIMEModel(nn.Module):
    """
    Unified VIME Model:
    Integrates Shared Encoder, Self-Supervised Pretext Heads (Mask Estimator + Feature Reconstructor),
    and Downstream Multi-Horizon Predictor Head.
    """
    def __init__(
        self,
        input_dim: int,
        target_dim: int = 8,
        latent_dim: int = 128,
        hidden_dim: int = 256,
        dropout: float = 0.1
    ):
        super(VIMEModel, self).__init__()
        self.input_dim = input_dim
        self.target_dim = target_dim
        self.latent_dim = latent_dim

        # Core components
        self.encoder = VIMEEncoder(input_dim=input_dim, latent_dim=latent_dim, hidden_dim=hidden_dim, dropout=dropout)
        self.mask_estimator = VIMEMaskEstimator(latent_dim=latent_dim, hidden_dim=hidden_dim // 2, output_dim=input_dim)
        self.reconstructor = VIMEFeatureReconstructor(latent_dim=latent_dim, hidden_dim=hidden_dim // 2, output_dim=input_dim)
        self.predictor = VIMEPredictor(latent_dim=latent_dim, hidden_dim=hidden_dim // 2, output_dim=target_dim, dropout=dropout)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encodes tabular input x into latent embedding z."""
        return self.encoder(x)

    def forward_pretrain(self, corrupted_x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Self-supervised pretext forward pass.
        Returns:
            mask_pred (torch.Tensor): Predicted mask probabilities in [0, 1]
            recon_x (torch.Tensor): Reconstructed continuous feature values
        """
        z = self.encoder(corrupted_x)
        mask_pred = self.mask_estimator(z)
        recon_x = self.reconstructor(z)
        return mask_pred, recon_x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Downstream prediction forward pass: x -> z -> y_hat.
        Returns:
            y_pred (torch.Tensor): Predicted displacements (e.g. 8 targets for 6h, 12h, 18h, 24h)
        """
        z = self.encoder(x)
        return self.predictor(z)

    def forward_with_recon(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes both prediction and reconstruction for unsupervised test adaptation.
        """
        z = self.encoder(x)
        y_pred = self.predictor(z)
        recon_x = self.reconstructor(z)
        return y_pred, recon_x
