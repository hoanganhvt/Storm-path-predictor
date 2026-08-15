import torch
from torch.utils.data import Dataset
from typing import List, Tuple
import numpy as np


class VariableStormDataset(Dataset):
    """
    PyTorch Dataset for Variable-Length Storm Sequences:
    - Inputs (X): list of (seq_len_i, num_features) historical storm states (at least 12 hours = 2 states)
    - Targets (Y): (N, 4, 2) 4 future coordinate displacements [[dlat_6h, dlon_6h], ..., [dlat_24h, dlon_24h]]
    - Coordinates: (N, 2) [current_lat, current_lon]
    """
    def __init__(
        self,
        X_list: List[np.ndarray],
        Y_arr: np.ndarray,
        coords_arr: np.ndarray,
        storm_ids: np.ndarray
    ):
        self.X_list = [torch.tensor(x, dtype=torch.float32) for x in X_list]
        self.Y = torch.tensor(Y_arr, dtype=torch.float32)
        self.coords = torch.tensor(coords_arr, dtype=torch.float32)
        self.storm_ids = storm_ids

    def __len__(self) -> int:
        return len(self.X_list)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.X_list[idx], self.Y[idx], self.coords[idx]


def left_pad_collate_fn(
    batch: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Collate function that left-pads variable-length storm sequences in a batch.
    Left-padding ensures that the last timestep (index -1) is always the latest/current observation.
    
    Returns
    -------
    padded_xs : torch.Tensor
        Batch of sequence tensors of shape (batch_size, max_seq_len_in_batch, num_features)
    ys : torch.Tensor
        Batch of target tensors of shape (batch_size, 4, 2)
    coords : torch.Tensor
        Batch of current storm coordinates [lat, lon] of shape (batch_size, 2)
    """
    xs, ys, coords = zip(*batch)
    max_len = max(x.size(0) for x in xs)
    feat_dim = xs[0].size(1)
    
    padded_xs = []
    for x in xs:
        seq_l = x.size(0)
        if seq_l < max_len:
            pad = torch.zeros(max_len - seq_l, feat_dim, dtype=x.dtype)
            padded_x = torch.cat([pad, x], dim=0)
        else:
            padded_x = x
        padded_xs.append(padded_x)
        
    return torch.stack(padded_xs, dim=0), torch.stack(ys, dim=0), torch.stack(coords, dim=0)
