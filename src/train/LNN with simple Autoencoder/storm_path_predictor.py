from typing import Dict, Optional, Tuple, Union
import torch
import torch.nn as nn

try:
    from .encoder import SimpleAutoencoderEncoder
    from .liquid_unet import CfCLiquidUNet1D
    from .heads import MultiHorizonFCHeads
except ImportError:
    from encoder import SimpleAutoencoderEncoder
    from liquid_unet import CfCLiquidUNet1D
    from heads import MultiHorizonFCHeads


class StormPathCfCLiquidUNet(nn.Module):
    """
    Storm Path Prediction Model (Liquid U-Net with Simple Autoencoder Embeddings):
    
    Modular Data Flow:
    1. Input Encoding (encoder.py): Raw storm features (B, T, input_dim) are encoded by 
       SimpleAutoencoderEncoder into an array of embedding vectors (B, T, latent_dim).
    2. Continuous-Time Liquid U-Net (liquid_unet.py): CfCLiquidUNet1D processes the sequence/array
       of embedding vectors across multiple temporal continuous-time scales.
    3. Multi-Horizon Forecasting (heads.py): MultiHorizonFCHeads outputs 4 horizons of (dlat, dlon).
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
        
        # 1. Simple Autoencoder Encoder (Stage 1: converts raw features to embedding vectors)
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
                
        # 2. CfC Liquid U-Net Core (Stage 2: receives array of embedding vectors)
        self.liquid_unet = CfCLiquidUNet1D(
            in_features=latent_dim,
            hidden_dims=unet_hidden_dims,
            backbone_units=cfc_backbone_units,
            backbone_layers=cfc_backbone_layers,
            dropout=dropout
        )
        
        # 3. Output Fully Connected Layers (Stage 3: 4 forecast heads x 2 displacement nodes)
        self.output_heads = MultiHorizonFCHeads(
            in_features=self.liquid_unet.out_dim,
            hidden_dim=fc_hidden_dim,
            num_horizons=num_horizons,
            nodes_per_horizon=nodes_per_horizon,
            dropout=dropout
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encodes raw storm features using the SimpleAutoencoderEncoder into an array of embedding vectors.
        
        Parameters
        ----------
        x : torch.Tensor
            Raw input storm features of shape (batch_size, seq_len, input_dim) 
            or (batch_size, input_dim).
            
        Returns
        -------
        torch.Tensor
            Array of embedding vectors of shape (batch_size, seq_len, latent_dim) 
            or (batch_size, latent_dim).
        """
        return self.encoder(x)

    def forward_embeddings(
        self,
        embeddings: torch.Tensor,
        timespans: Optional[torch.Tensor] = None,
        return_dict: bool = False
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Feeds an array of embedding vectors directly into the Liquid U-Net and output heads.
        
        Parameters
        ----------
        embeddings : torch.Tensor
            Array of embedding vectors of shape (batch_size, seq_len, latent_dim).
        timespans : Optional[torch.Tensor]
            Optional time elapsed between time steps for continuous-time ODE dynamics.
        return_dict : bool
            If True, returns a dictionary containing named outputs for each horizon.
            
        Returns
        -------
        torch.Tensor or Dict[str, torch.Tensor]
            - Tensor of shape (batch_size, num_horizons, 2)
            - Or Dict with keys ['6h', '12h', '18h', '24h', 'all']
        """
        # Pass array of embedding vectors through Liquid U-Net Core
        unet_seq_features = self.liquid_unet(embeddings, timespans=timespans)
        
        # Extract latest dynamic state
        core_repr = self.liquid_unet.extract_latest_state(unet_seq_features)
        
        # Generate 4-horizon trajectory predictions
        stacked_preds, head_preds = self.output_heads(core_repr)
        
        if return_dict:
            horizon_names = [f"{(i+1)*6}h" for i in range(self.num_horizons)]
            result = {name: head_preds[i] for i, name in enumerate(horizon_names)}
            result['all'] = stacked_preds
            return result
            
        return stacked_preds

    def forward(
        self,
        x: torch.Tensor,
        timespans: Optional[torch.Tensor] = None,
        is_pre_encoded: bool = False,
        return_dict: bool = False
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Full forward pass for the Storm Path Predictor.
        
        Workflow:
        1. Encode input sequence 'x' via Autoencoder to get array of embedding vectors.
        2. Pass embedding array into CfC Liquid U-Net Core.
        3. Pass extracted core state through Multi-Horizon FC Heads.
        
        Parameters
        ----------
        x : torch.Tensor
            Raw storm trajectory input of shape (batch_size, seq_len, input_dim)
            or pre-encoded embedding array if is_pre_encoded=True.
        timespans : Optional[torch.Tensor]
            Time intervals between steps for CfC continuous-time dynamics.
        is_pre_encoded : bool
            If True, 'x' is already an array of embedding vectors.
        return_dict : bool
            If True, returns a dict with predictions per horizon.
            
        Returns
        -------
        torch.Tensor or Dict[str, torch.Tensor]
            Prediction tensor (batch_size, 4, 2) or dictionary.
        """
        # Ensure 3D shape (batch_size, seq_len, dim)
        if x.dim() == 2:
            x = x.unsqueeze(1)
            
        # Step 1: Obtain array of embedding vectors
        if is_pre_encoded:
            embeddings = x
        else:
            embeddings = self.encode(x)
            
        # Step 2 & 3: Pass embedding array through Liquid UNet and output heads
        return self.forward_embeddings(embeddings, timespans=timespans, return_dict=return_dict)


# Semantic alias
LiquidUNetWithAutoencoder = StormPathCfCLiquidUNet


if __name__ == "__main__":
    print("Testing StormPathCfCLiquidUNet (Composite Predictor)...")
    model = StormPathCfCLiquidUNet(input_dim=15, latent_dim=64)
    raw_x = torch.randn(8, 6, 15)
    
    # 1. Test encoding
    embeds = model.encode(raw_x)
    print(f"Raw input: {raw_x.shape} --> Embeddings array: {embeds.shape}")
    assert embeds.shape == (8, 6, 64)
    
    # 2. Test forward_embeddings
    preds_embeds = model.forward_embeddings(embeds)
    print(f"Forward embeddings output: {preds_embeds.shape}")
    assert preds_embeds.shape == (8, 4, 2)
    
    # 3. Test end-to-end
    model.eval()
    with torch.no_grad():
        preds_raw = model(raw_x)
        assert preds_raw.shape == (8, 4, 2)
        print("End-to-end forward test passed!")
