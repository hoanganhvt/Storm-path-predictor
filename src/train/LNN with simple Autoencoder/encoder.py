import os
from typing import Optional
import torch
import torch.nn as nn


class SimpleAutoencoderEncoder(nn.Module):
    """
    Encoder module matching the exact architecture of the SimpleAutoencoder.
    
    Transforms raw tabular storm features (e.g., 15-dimensional) at each time step
    into a compact latent embedding vector (e.g., 64-dimensional).
    """
    def __init__(self, input_dim: int = 15, latent_dim: int = 64):
        super(SimpleAutoencoderEncoder, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim),
            nn.ReLU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for encoder. Encodes raw input into latent space.
        
        Parameters
        ----------
        x : torch.Tensor
            Input storm features of shape (batch_size, input_dim) 
            or sequence of features (batch_size, seq_len, input_dim).
            
        Returns
        -------
        torch.Tensor
            Latent embedding vector of shape (batch_size, latent_dim) 
            or array of embedding vectors of shape (batch_size, seq_len, latent_dim).
        """
        return self.encoder(x)

    def load_pretrained_weights(self, checkpoint_path: str, strict: bool = False) -> None:
        """
        Loads encoder weights from a saved SimpleAutoencoder checkpoint file.
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found at: '{checkpoint_path}'")
            
        ckpt = torch.load(checkpoint_path, map_location='cpu')
        state_dict = ckpt['model_state_dict'] if isinstance(ckpt, dict) and 'model_state_dict' in ckpt else ckpt
        
        # Filter for encoder keys only
        encoder_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('encoder.'):
                encoder_state_dict[k] = v
                
        if not encoder_state_dict:
            raise KeyError(f"No encoder weights found in checkpoint '{checkpoint_path}'.")
            
        msg = self.load_state_dict(encoder_state_dict, strict=strict)
        print(f"[SUCCESS] Loaded pretrained SimpleAutoencoder encoder weights from '{checkpoint_path}': {msg}")

    def freeze(self, freeze: bool = True) -> None:
        """Freezes or unfreezes encoder parameters."""
        for param in self.parameters():
            param.requires_grad = not freeze


if __name__ == "__main__":
    print("Testing SimpleAutoencoderEncoder...")
    encoder = SimpleAutoencoderEncoder(input_dim=15, latent_dim=64)
    sample_input = torch.randn(8, 6, 15)
    embeddings = encoder(sample_input)
    print(f"Input shape: {sample_input.shape} --> Embeddings array shape: {embeddings.shape}")
    assert embeddings.shape == (8, 6, 64)
    print("SimpleAutoencoderEncoder test passed!")
