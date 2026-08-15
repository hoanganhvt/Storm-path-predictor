"""
Central Model Module for Storm Path Prediction (LNN with Simple Autoencoder).

Modular Architecture:
- SimpleAutoencoderEncoder: Defined in `encoder.py` (Tabular feature embedding)
- CfCLiquidUNet1D: Defined in `liquid_unet.py` (Continuous-time Liquid U-Net core)
- MultiHorizonFCHeads: Defined in `heads.py` (Multi-horizon displacement forecast heads)
- StormPathCfCLiquidUNet: Defined in `storm_path_predictor.py` (Composite end-to-end predictor)
"""

try:
    from .encoder import SimpleAutoencoderEncoder
    from .liquid_unet import CfCLiquidUNet1D
    from .heads import MultiHorizonFCHeads
    from .storm_path_predictor import StormPathCfCLiquidUNet, LiquidUNetWithAutoencoder
except ImportError:
    from encoder import SimpleAutoencoderEncoder
    from liquid_unet import CfCLiquidUNet1D
    from heads import MultiHorizonFCHeads
    from storm_path_predictor import StormPathCfCLiquidUNet, LiquidUNetWithAutoencoder

__all__ = [
    "SimpleAutoencoderEncoder",
    "CfCLiquidUNet1D",
    "MultiHorizonFCHeads",
    "StormPathCfCLiquidUNet",
    "LiquidUNetWithAutoencoder",
]


if __name__ == "__main__":
    import torch
    print("=" * 80)
    print("TESTING MODULAR MODEL COMPONENTS EXPORTED VIA model.py")
    print("=" * 80)
    
    # Verify instantiation of each modular component
    enc = SimpleAutoencoderEncoder(input_dim=15, latent_dim=64)
    unet = CfCLiquidUNet1D(in_features=64, hidden_dims=(64, 128, 256))
    heads = MultiHorizonFCHeads(in_features=64, hidden_dim=128, num_horizons=4, nodes_per_horizon=2)
    predictor = StormPathCfCLiquidUNet(input_dim=15, latent_dim=64)
    
    x = torch.randn(4, 5, 15)
    e = enc(x)
    u = unet(e)
    h_out, _ = heads(unet.extract_latest_state(u))
    p_out = predictor(x)
    
    print(f"1. Encoder: Raw {tuple(x.shape)} -> Embedding Array {tuple(e.shape)} [OK]")
    print(f"2. Liquid UNet: Embeddings {tuple(e.shape)} -> Sequence Features {tuple(u.shape)} [OK]")
    print(f"3. Heads: Core State (4, 64) -> 4 Horizons {tuple(h_out.shape)} [OK]")
    print(f"4. Full Predictor: Raw {tuple(x.shape)} -> Predictions {tuple(p_out.shape)} [OK]")
    print("\nAll modular model components verified successfully!")
