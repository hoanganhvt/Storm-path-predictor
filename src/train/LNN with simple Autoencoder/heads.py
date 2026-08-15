from typing import List, Tuple
import torch
import torch.nn as nn


class MultiHorizonFCHeads(nn.Module):
    """
    Fully Connected Output Block with 4 prediction heads, each outputting 2 nodes.
    
    Structure:
    - Shared Multi-Layer Perceptron (MLP) layers with LayerNorm, ReLU, and Dropout.
    - 4 distinct output heads (e.g. 6h, 12h, 18h, 24h horizons), each producing 2 nodes
      (e.g., [delta_lat, delta_lon]).
    """
    def __init__(
        self,
        in_features: int = 64,
        hidden_dim: int = 128,
        num_horizons: int = 4,
        nodes_per_horizon: int = 2,
        dropout: float = 0.1
    ):
        super(MultiHorizonFCHeads, self).__init__()
        self.num_horizons = num_horizons
        self.nodes_per_horizon = nodes_per_horizon
        
        # Shared feature refinement MLP
        self.shared_fc = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        head_in = hidden_dim // 2
        
        # 4 distinct prediction heads, each containing fully connected layers outputting 2 nodes
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(head_in, 32),
                nn.ReLU(),
                nn.Linear(32, nodes_per_horizon)
            )
            for _ in range(num_horizons)
        ])

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass for the 4 FC output heads.
        
        Parameters
        ----------
        x : torch.Tensor
            Aggregated core representation of shape (batch_size, in_features).
            
        Returns
        -------
        stacked_outputs : torch.Tensor
            Tensor of shape (batch_size, num_horizons, nodes_per_horizon) containing all horizons.
        head_outputs : List[torch.Tensor]
            List of tensors, each of shape (batch_size, nodes_per_horizon).
        """
        feat = self.shared_fc(x)
        head_outputs = [head(feat) for head in self.heads]  # 4 x (batch_size, 2)
        stacked_outputs = torch.stack(head_outputs, dim=1)  # (batch_size, 4, 2)
        return stacked_outputs, head_outputs


if __name__ == "__main__":
    print("Testing MultiHorizonFCHeads...")
    heads = MultiHorizonFCHeads(in_features=64, hidden_dim=128, num_horizons=4, nodes_per_horizon=2)
    sample_feat = torch.randn(8, 64)
    stacked, separate = heads(sample_feat)
    print(f"Input feat shape: {sample_feat.shape} --> Stacked heads output shape: {stacked.shape}")
    assert stacked.shape == (8, 4, 2)
    assert len(separate) == 4 and separate[0].shape == (8, 2)
    print("MultiHorizonFCHeads test passed!")
