import os
from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from ncps.torch import CfC
except ImportError:
    raise ImportError("ncps package is required. Please install it with `pip install ncps`.")


class SimpleAutoencoderEncoder(nn.Module):
    """
    Encoder module matching the exact architecture of the SimpleAutoencoder.
    
    Transforms raw tabular storm features (e.g., 15-dimensional) into a compact
    latent representation (e.g., 64-dimensional).
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
            or (batch_size, seq_len, input_dim).
            
        Returns
        -------
        torch.Tensor
            Latent features of shape (batch_size, latent_dim) 
            or (batch_size, seq_len, latent_dim).
        """
        return self.encoder(x)

    def load_pretrained_weights(self, checkpoint_path: str, strict: bool = False) -> None:
        """
        Loads encoder weights from a saved SimpleAutoencoder checkpoint file.
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found at: '{checkpoint_path}'")
            
        ckpt = torch.load(checkpoint_path, map_location='cpu')
        state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
        
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


class CfCLiquidUNet1D(nn.Module):
    """
    1D Continuous-Time Liquid U-Net Core using Closed-form Continuous-time (CfC) layers.
    
    Architecture:
    - Contracting path (Encoder): Multi-scale temporal feature extraction with CfC blocks and downsampling.
    - Bottleneck: Deep continuous-time CfC layer capturing long-range storm dynamics.
    - Expanding path (Decoder): Temporal upsampling, multi-scale skip connections, and CfC synthesis blocks.
    """
    def __init__(
        self,
        in_features: int = 64,
        hidden_dims: Tuple[int, int, int] = (64, 128, 256),
        backbone_units: int = 64,
        backbone_layers: int = 1,
        mode: str = "default",
        dropout: float = 0.1
    ):
        super(CfCLiquidUNet1D, self).__init__()
        c1, c2, c3 = hidden_dims
        self.in_features = in_features
        self.hidden_dims = hidden_dims
        
        # =========================================================================
        # 1. Contracting Path (Encoder)
        # =========================================================================
        # Level 1
        self.enc_cfc1 = CfC(
            in_features, c1, 
            batch_first=True, return_sequences=True,
            backbone_units=backbone_units, backbone_layers=backbone_layers, mode=mode
        )
        self.enc_norm1 = nn.LayerNorm(c1)
        self.down1 = nn.Conv1d(c1, c2, kernel_size=3, stride=2, padding=1)
        
        # Level 2
        self.enc_cfc2 = CfC(
            c2, c2, 
            batch_first=True, return_sequences=True,
            backbone_units=backbone_units, backbone_layers=backbone_layers, mode=mode
        )
        self.enc_norm2 = nn.LayerNorm(c2)
        self.down2 = nn.Conv1d(c2, c3, kernel_size=3, stride=2, padding=1)
        
        # =========================================================================
        # 2. Bottleneck (Deep Liquid Core)
        # =========================================================================
        self.bottleneck_cfc = CfC(
            c3, c3, 
            batch_first=True, return_sequences=True,
            backbone_units=backbone_units, backbone_layers=backbone_layers, mode=mode
        )
        self.bottleneck_norm = nn.LayerNorm(c3)
        self.dropout = nn.Dropout(dropout)
        
        # =========================================================================
        # 3. Expanding Path (Decoder with Skip Connections)
        # =========================================================================
        # Level 2 Decoder
        self.up2_proj = nn.Conv1d(c3, c2, kernel_size=1)
        self.dec_cfc2 = CfC(
            c2 + c2, c2, 
            batch_first=True, return_sequences=True,
            backbone_units=backbone_units, backbone_layers=backbone_layers, mode=mode
        )
        self.dec_norm2 = nn.LayerNorm(c2)
        
        # Level 1 Decoder
        self.up1_proj = nn.Conv1d(c2, c1, kernel_size=1)
        self.dec_cfc1 = CfC(
            c1 + c1, c1, 
            batch_first=True, return_sequences=True,
            backbone_units=backbone_units, backbone_layers=backbone_layers, mode=mode
        )
        self.dec_norm1 = nn.LayerNorm(c1)
        
        self.out_dim = c1

    def forward(self, x: torch.Tensor, timespans: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through the CfC Liquid U-Net Core.
        
        Parameters
        ----------
        x : torch.Tensor
            Latent sequence tensor of shape (batch_size, seq_len, in_features).
        timespans : Optional[torch.Tensor]
            Optional time elapsed between time steps for continuous-time ODE dynamics.
            
        Returns
        -------
        torch.Tensor
            U-Net processed sequence tensor of shape (batch_size, seq_len, out_dim).
        """
        # -------------------------------------------------------------
        # Level 1 Encoder
        # -------------------------------------------------------------
        e1, _ = self.enc_cfc1(x, timespans=timespans)
        e1 = self.enc_norm1(F.relu(e1))  # (B, T, c1)
        
        # Downsample Level 1 -> Level 2
        e1_t = e1.transpose(1, 2)       # (B, c1, T)
        d1 = self.down1(e1_t).transpose(1, 2)  # (B, T_down1, c2)
        
        # -------------------------------------------------------------
        # Level 2 Encoder
        # -------------------------------------------------------------
        e2, _ = self.enc_cfc2(d1)
        e2 = self.enc_norm2(F.relu(e2))  # (B, T_down1, c2)
        
        # Downsample Level 2 -> Bottleneck
        e2_t = e2.transpose(1, 2)       # (B, c2, T_down1)
        d2 = self.down2(e2_t).transpose(1, 2)  # (B, T_down2, c3)
        
        # -------------------------------------------------------------
        # Bottleneck (Deep Liquid Core)
        # -------------------------------------------------------------
        b, _ = self.bottleneck_cfc(d2)
        b = self.bottleneck_norm(F.relu(b))
        b = self.dropout(b)             # (B, T_down2, c3)
        
        # -------------------------------------------------------------
        # Level 2 Decoder (with Skip Connection from e2)
        # -------------------------------------------------------------
        b_t = b.transpose(1, 2)         # (B, c3, T_down2)
        # Upsample temporal dimension to match e2's sequence length
        u2 = F.interpolate(b_t, size=e2.size(1), mode='nearest')
        u2 = self.up2_proj(u2).transpose(1, 2)  # (B, T_down1, c2)
        
        # Skip Connection: Concatenate along feature dimension
        cat2 = torch.cat([u2, e2], dim=-1)      # (B, T_down1, c2 + c2)
        d_out2, _ = self.dec_cfc2(cat2)
        d_out2 = self.dec_norm2(F.relu(d_out2)) # (B, T_down1, c2)
        
        # -------------------------------------------------------------
        # Level 1 Decoder (with Skip Connection from e1)
        # -------------------------------------------------------------
        d_out2_t = d_out2.transpose(1, 2)       # (B, c2, T_down1)
        # Upsample temporal dimension to match original sequence length T
        u1 = F.interpolate(d_out2_t, size=e1.size(1), mode='nearest')
        u1 = self.up1_proj(u1).transpose(1, 2)  # (B, T, c1)
        
        # Skip Connection: Concatenate along feature dimension
        cat1 = torch.cat([u1, e1], dim=-1)      # (B, T, c1 + c1)
        d_out1, _ = self.dec_cfc1(cat1)
        d_out1 = self.dec_norm1(F.relu(d_out1)) # (B, T, c1)
        
        return d_out1


class MultiHorizonFCHeads(nn.Module):
    """
    Fully Connected Output Block with 4 prediction heads, each outputting 2 nodes.
    
    Structure:
    - Shared Multi-Layer Perceptron (MLP) layers with LayerNorm, ReLU, and Dropout.
    - 4 distinct output heads (e.g. 6h, 12h, 18h, 24h horizons), each producing 2 nodes (e.g., [lat, lon] or [delta_lat, delta_lon]).
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
            Tensor of shape (batch_size, 4, 2) containing all 4 horizons.
        head_outputs : List[torch.Tensor]
            List of 4 tensors, each of shape (batch_size, 2).
        """
        feat = self.shared_fc(x)
        head_outputs = [head(feat) for head in self.heads]  # 4 x (batch_size, 2)
        stacked_outputs = torch.stack(head_outputs, dim=1)  # (batch_size, 4, 2)
        return stacked_outputs, head_outputs


class StormPathCfCLiquidUNet(nn.Module):
    """
    Storm Path Prediction Model:
    1. Input Encoding: Raw storm features are encoded FIRST with SimpleAutoencoderEncoder into latent space.
    2. Continuous-Time Core: CfC Liquid U-Net processes the sequence of latent representations.
    3. Multi-Head Output: Fully connected layers split into 4 horizon heads, each outputting 2 nodes.
    """
    def __init__(
        self,
        input_dim: int = 15,
        latent_dim: int = 64,
        encoder: Optional[Union[SimpleAutoencoderEncoder, nn.Module]] = None,
        unet_hidden_dims: Tuple[int, int, int] = (64, 128, 256),
        fc_hidden_dim: int = 128,
        num_horizons: int = 4,
        nodes_per_horizon: int = 2,
        cfc_backbone_units: int = 64,
        cfc_backbone_layers: int = 1,
        dropout: float = 0.1,
        pretrained_autoencoder_path: Optional[str] = None,
        freeze_encoder: bool = False
    ):
        super(StormPathCfCLiquidUNet, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.num_horizons = num_horizons
        self.nodes_per_horizon = nodes_per_horizon
        
        # 1. Simple Autoencoder Encoder (First stage: input encoding)
        if encoder is not None:
            self.encoder = encoder
        else:
            self.encoder = SimpleAutoencoderEncoder(input_dim=input_dim, latent_dim=latent_dim)
        
        # Optionally load pretrained encoder checkpoint
        if pretrained_autoencoder_path:
            self.encoder.load_pretrained_weights(pretrained_autoencoder_path)
            if freeze_encoder:
                self.encoder.freeze(True)
                print("[INFO] Pretrained Autoencoder Encoder frozen.")
                
        # 2. CfC Liquid U-Net Core (Second stage: dynamic spatio-temporal modeling)
        self.liquid_unet = CfCLiquidUNet1D(
            in_features=latent_dim,
            hidden_dims=unet_hidden_dims,
            backbone_units=cfc_backbone_units,
            backbone_layers=cfc_backbone_layers,
            dropout=dropout
        )
        
        # 3. Output Fully Connected Layers (Third stage: 4 heads x 2 nodes)
        self.output_heads = MultiHorizonFCHeads(
            in_features=self.liquid_unet.out_dim,
            hidden_dim=fc_hidden_dim,
            num_horizons=num_horizons,
            nodes_per_horizon=nodes_per_horizon,
            dropout=dropout
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Explicitly encodes the input storm features using the SimpleAutoencoderEncoder FIRST.
        
        Parameters
        ----------
        x : torch.Tensor
            Raw input storm features. Shape: (batch_size, seq_len, input_dim) or (batch_size, input_dim).
            
        Returns
        -------
        torch.Tensor
            Latent features. Shape: (batch_size, seq_len, latent_dim) or (batch_size, latent_dim).
        """
        return self.encoder(x)

    def forward(
        self,
        x: torch.Tensor,
        timespans: Optional[torch.Tensor] = None,
        is_pre_encoded: bool = False,
        return_dict: bool = False
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass for the full Storm Path Predictor.
        
        Workflow:
        Step 1: The input features 'x' are encoded FIRST with SimpleAutoencoderEncoder.
        Step 2: The latent representations are passed into the CfC Liquid U-Net Core.
        Step 3: The core representation is passed to the 4 fully connected output heads (2 nodes each).
        
        Parameters
        ----------
        x : torch.Tensor
            Input storm trajectory data (or pre-encoded latents if is_pre_encoded=True).
            Shape: (batch_size, seq_len, input_dim) or (batch_size, input_dim).
        timespans : Optional[torch.Tensor]
            Time intervals between steps for CfC continuous-time dynamics.
        is_pre_encoded : bool
            If True, 'x' is assumed to already be encoded latent vectors (skips Step 1).
        return_dict : bool
            If True, returns a dictionary containing named outputs for each horizon.
            
        Returns
        -------
        torch.Tensor or Dict[str, torch.Tensor]
            - By default: Tensor of shape (batch_size, 4, 2)
            - If return_dict=True: Dict with keys ['6h', '12h', '18h', '24h', 'all']
        """
        # Handle 2D input (batch_size, input_dim) by adding sequence dimension (batch_size, 1, input_dim)
        if x.dim() == 2:
            x = x.unsqueeze(1)
            
        # =========================================================================
        # STEP 1: Encode raw inputs FIRST with SimpleAutoencoderEncoder
        # (batch_size, seq_len, input_dim) -> (batch_size, seq_len, latent_dim)
        # =========================================================================
        if is_pre_encoded:
            latents = x
        else:
            latents = self.encode(x)
        
        # =========================================================================
        # STEP 2: Pass encoded latents through CfC Liquid U-Net Core
        # (batch_size, seq_len, latent_dim) -> (batch_size, seq_len, c_out)
        # =========================================================================
        unet_seq_features = self.liquid_unet(latents, timespans=timespans)
        
        # Extract the latest temporal state (or final sequence step)
        core_repr = unet_seq_features[:, -1, :]  # (batch_size, c_out)
        
        # =========================================================================
        # STEP 3: Pass through Fully Connected Layers into 4 outputs of 2 nodes each
        # (batch_size, c_out) -> 4 x (batch_size, 2) -> (batch_size, 4, 2)
        # =========================================================================
        stacked_preds, head_preds = self.output_heads(core_repr)
        
        if return_dict:
            horizon_names = [f"{(i+1)*6}h" for i in range(self.num_horizons)]
            result = {name: head_preds[i] for i, name in enumerate(horizon_names)}
            result['all'] = stacked_preds
            return result
            
        return stacked_preds


if __name__ == "__main__":
    print("=" * 70)
    print("Testing StormPathCfCLiquidUNet Model Declaration (First Encoded with SimpleAutoencoderEncoder)...")
    print("=" * 70)
    
    # 1. Initialize model
    model = StormPathCfCLiquidUNet(
        input_dim=15,
        latent_dim=64,
        unet_hidden_dims=(64, 128, 256),
        fc_hidden_dim=128,
        num_horizons=4,
        nodes_per_horizon=2,
        dropout=0.1
    )
    
    print("\nModel Architecture Summary:")
    print(model)
    
    # 2. Test Step 1: Explicit Input Encoding
    batch_size = 8
    seq_len = 6
    input_dim = 15
    raw_storm_input = torch.randn(batch_size, seq_len, input_dim)
    
    print(f"\n1. Testing explicit encoding step on raw input: {raw_storm_input.shape}...")
    encoded_latents = model.encode(raw_storm_input)
    print(f"   --> Encoded latents shape: {encoded_latents.shape} (Expected: [{batch_size}, {seq_len}, 64])")
    assert encoded_latents.shape == (batch_size, seq_len, 64)
    
    # 3. Test Full Forward Pass (Raw inputs -> Encoder -> CfC Liquid UNet -> 4 FC outputs of 2 nodes)
    model.eval()
    with torch.no_grad():
        print(f"\n2. Testing end-to-end forward pass (raw input -> encoder first -> CfC UNet -> FC heads)...")
        output_tensor = model(raw_storm_input)
        print(f"   --> Output Tensor Shape: {output_tensor.shape} (Expected: [{batch_size}, 4, 2])")
        assert output_tensor.shape == (batch_size, 4, 2)
        
        # 4. Test Forward Pass with Pre-Encoded Latents (is_pre_encoded=True)
        print(f"\n3. Testing forward pass with pre-encoded latents (is_pre_encoded=True)...")
        output_from_latents = model(encoded_latents, is_pre_encoded=True)
        print(f"   --> Output from latents shape: {output_from_latents.shape} (Expected: [{batch_size}, 4, 2])")
        assert torch.allclose(output_tensor, output_from_latents, atol=1e-5)
        print("   --> Outputs match perfectly between raw and pre-encoded pathways!")
    
    # 5. Test loading pretrained autoencoder weights if checkpoint exists
    sample_ckpt_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../Simple Autoencoder/checkpoints/best_autoencoder.pth")
    )
    if os.path.exists(sample_ckpt_path):
        print(f"\n4. Testing pretrained autoencoder weight loading from: '{sample_ckpt_path}'...")
        pretrained_model = StormPathCfCLiquidUNet(
            input_dim=15,
            latent_dim=64,
            pretrained_autoencoder_path=sample_ckpt_path,
            freeze_encoder=True
        )
        out = pretrained_model(raw_storm_input)
        print(f"   --> Pretrained model forward pass successful! Output shape: {out.shape}")
        
    print("\nAll model tests passed successfully!")
