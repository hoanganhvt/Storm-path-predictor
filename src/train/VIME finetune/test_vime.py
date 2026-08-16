import os
import sys
import unittest
import numpy as np
import torch

from vime_model import (
    VIMEEncoder,
    VIMEMaskEstimator,
    VIMEFeatureReconstructor,
    VIMEPredictor,
    VIMEModel
)
from vime_utils import (
    haversine_np,
    compute_cosine_similarity,
    VIMECorruptor,
    load_and_preprocess_storm_data,
    evaluate_storm_predictions
)


class TestVIMEPipeline(unittest.TestCase):
    def setUp(self):
        self.batch_size = 16
        self.input_dim = 40
        self.target_dim = 8
        self.latent_dim = 64
        self.hidden_dim = 128
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def test_haversine_formula(self):
        """Test Haversine distance formula against known geographic distance."""
        # Tokyo (35.6762° N, 139.6503° E) to Osaka (34.6937° N, 135.5023° E) is ~396-405 km
        tokyo_lat, tokyo_lon = np.array([35.6762]), np.array([139.6503])
        osaka_lat, osaka_lon = np.array([34.6937]), np.array([135.5023])
        
        dist = haversine_np(tokyo_lon, tokyo_lat, osaka_lon, osaka_lat)
        self.assertAlmostEqual(dist[0], 400.0, delta=20.0)

    def test_cosine_similarity(self):
        """Test directional cosine similarity for displacement vectors."""
        v1_lat, v1_lon = np.array([1.0, 0.0]), np.array([0.0, 1.0])
        v2_lat, v2_lon = np.array([1.0, 0.0]), np.array([0.0, -1.0])
        
        sim, arr = compute_cosine_similarity(v1_lat, v1_lon, v2_lat, v2_lon)
        self.assertAlmostEqual(arr[0], 1.0, places=5)
        self.assertAlmostEqual(arr[1], -1.0, places=5)
        self.assertAlmostEqual(sim, 0.0, places=5)

    def test_vime_corruptor(self):
        """Test empirical marginal corruptor and K-consistency perturbation generator."""
        bg_data = np.random.randn(100, self.input_dim).astype(np.float32)
        corruptor = VIMECorruptor(bg_data, p_m=0.4)
        
        x = torch.randn(self.batch_size, self.input_dim)
        corrupted_x, mask = corruptor.corrupt(x)
        
        self.assertEqual(corrupted_x.shape, (self.batch_size, self.input_dim))
        self.assertEqual(mask.shape, (self.batch_size, self.input_dim))
        self.assertTrue(torch.all((mask == 0.0) | (mask == 1.0)))

        # Test K corruptions
        K = 4
        corrupted_k = corruptor.generate_k_corruptions(x, K=K)
        self.assertEqual(corrupted_k.shape, (K, self.batch_size, self.input_dim))

    def test_model_forward_and_backward(self):
        """Test VIME model forward passes, pretext losses, and backward propagation."""
        model = VIMEModel(
            input_dim=self.input_dim,
            target_dim=self.target_dim,
            latent_dim=self.latent_dim,
            hidden_dim=self.hidden_dim
        ).to(self.device)

        x = torch.randn(self.batch_size, self.input_dim).to(self.device)
        y = torch.randn(self.batch_size, self.target_dim).to(self.device)
        mask = (torch.rand(self.batch_size, self.input_dim, device=self.device) < 0.3).float()

        # 1. Pretext forward
        mask_pred, recon_x = model.forward_pretrain(x)
        self.assertEqual(mask_pred.shape, (self.batch_size, self.input_dim))
        self.assertEqual(recon_x.shape, (self.batch_size, self.input_dim))
        self.assertTrue(torch.all(mask_pred >= 0.0) and torch.all(mask_pred <= 1.0))

        # 2. Downstream prediction forward
        y_pred = model(x)
        self.assertEqual(y_pred.shape, (self.batch_size, self.target_dim))

        # 3. Combined loss and backprop
        loss_sup = torch.nn.functional.mse_loss(y_pred, y)
        loss_mask = torch.nn.functional.binary_cross_entropy(mask_pred, mask)
        loss_recon = torch.nn.functional.mse_loss(recon_x, x)
        
        total_loss = loss_sup + 0.5 * loss_mask + 0.5 * loss_recon
        total_loss.backward()

        # Check gradients exist
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad, f"Gradient is None for {name}")

    def test_data_pipeline(self):
        """Test real dataset feature engineering and preprocessing pipeline."""
        train_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../Datasets/Storm_data/train.csv'))
        test_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../Datasets/Storm_data/test.csv'))

        if os.path.exists(train_path) and os.path.exists(test_path):
            data_dict = load_and_preprocess_storm_data(train_path, test_path)
            self.assertIn('X_train', data_dict)
            self.assertIn('y_train', data_dict)
            self.assertIn('X_test', data_dict)
            self.assertIn('y_test', data_dict)
            self.assertEqual(data_dict['y_train'].shape[1], 8)
            self.assertEqual(data_dict['y_test'].shape[1], 8)
            self.assertFalse(np.isnan(data_dict['X_train']).any(), "X_train contains NaNs!")
            self.assertFalse(np.isnan(data_dict['X_test']).any(), "X_test contains NaNs!")
            print(f"\n[PASS] Verified real data shapes: Train={data_dict['X_train'].shape}, Test={data_dict['X_test'].shape}")


if __name__ == '__main__':
    unittest.main()
