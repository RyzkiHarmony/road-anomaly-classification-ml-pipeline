import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(_PROJECT_ROOT, "src", "utils"))
from data_utils import get_stratified_group_split


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def scale_instance_level(X, eps=1e-8):
    """
    Standardize each sample independently across its time dimension:
    X: shape (N, C, T) or (C, T)
    """
    if X.ndim == 3:
        mean = X.mean(axis=-1, keepdims=True)
        std = X.std(axis=-1, keepdims=True)
        return (X - mean) / (std + eps)
    elif X.ndim == 2:
        mean = X.mean(axis=-1, keepdims=True)
        std = X.std(axis=-1, keepdims=True)
        return (X - mean) / (std + eps)
    return X


class DynamicJitterDataset(torch.utils.data.Dataset):
    def __init__(self, X, y, max_jitter=15, noise_std=0.02, scale_range=(0.85, 1.15),
                 time_warp_prob=0.8, time_warp_mag=0.1, channel_drop_prob=0.1,
                 is_train=True):
        self.X = X
        self.y = y
        self.max_jitter = max_jitter
        self.noise_std = noise_std
        self.scale_range = scale_range
        self.time_warp_prob = time_warp_prob
        self.time_warp_mag = time_warp_mag
        self.channel_drop_prob = channel_drop_prob
        self.is_train = is_train

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx].clone()  # Clone to avoid in-place modification
        y_val = self.y[idx]
        if self.is_train:
            # 1. Random Crop (Replaces padding-based temporal jitter)
            seq_len = 200
            if x.shape[-1] > seq_len:
                max_start_idx = x.shape[-1] - seq_len
                start_idx = np.random.randint(0, max_start_idx + 1) if self.max_jitter > 0 else max_start_idx // 2
                x = x[..., start_idx:start_idx + seq_len]
            elif x.shape[-1] == seq_len:
                pass
            else:
                raise ValueError(f"Input sequence length {x.shape[-1]} is shorter than target length {seq_len}")

            # 2. Time Warping (non-linear temporal deformation)
            if self.time_warp_prob > 0 and np.random.rand() < self.time_warp_prob:
                T = x.shape[-1]
                n_knots = 4
                knot_positions = np.linspace(0, T - 1, n_knots + 2)
                knot_offsets = np.random.uniform(-self.time_warp_mag * T,
                                                  self.time_warp_mag * T,
                                                  size=n_knots + 2)
                knot_offsets[0] = 0  # Anchor start
                knot_offsets[-1] = 0  # Anchor end

                orig_indices = np.arange(T, dtype=np.float32)
                warped_indices = np.interp(orig_indices, knot_positions,
                                           knot_positions + knot_offsets)
                warped_indices = np.clip(warped_indices, 0, T - 1)

                warped_int = warped_indices.astype(np.int64)
                warped_frac = warped_indices - warped_int
                warped_int_next = np.minimum(warped_int + 1, T - 1)

                warped_frac_t = torch.from_numpy(warped_frac).float().unsqueeze(0)
                x = x[:, warped_int] * (1 - warped_frac_t) + x[:, warped_int_next] * warped_frac_t

            # 3. Gaussian Noise Injection
            if self.noise_std > 0 and np.random.rand() < 1.0:
                noise = torch.randn_like(x) * self.noise_std
                x = x + noise

            # 4. Magnitude Scaling (per-channel random scale)
            if self.scale_range is not None and np.random.rand() < 1.0:
                lo, hi = self.scale_range
                n_channels = x.shape[0]
                scale = torch.FloatTensor(n_channels, 1).uniform_(lo, hi)
                x = x * scale

            # 5. Channel Dropout (zero out a random channel)
            if self.channel_drop_prob > 0 and np.random.rand() < self.channel_drop_prob:
                n_channels = x.shape[0]
                drop_idx = np.random.randint(0, n_channels)
                x[drop_idx, :] = 0.0

        else:
            # During evaluation, strictly use center crop
            seq_len = 200
            if x.shape[-1] > seq_len:
                start_idx = (x.shape[-1] - seq_len) // 2
                x = x[..., start_idx:start_idx + seq_len]

        return x, y_val


class MultiClassFocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0, reduction='mean'):
        super(MultiClassFocalLoss, self).__init__()
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # Calculate raw CE loss WITHOUT weights to get correct pt mathematically
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none', weight=None)
        pt = torch.exp(-ce_loss)  # probability of correct prediction
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        # Apply class weights manually if provided
        if self.weight is not None:
            target_weights = self.weight[targets]
            focal_loss = focal_loss * target_weights

        if self.reduction == 'mean':
            if self.weight is not None:
                return focal_loss.sum() / target_weights.sum()
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


class MultiLabelFocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0, reduction='mean'):
        super(MultiLabelFocalLoss, self).__init__()
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        bce_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none', weight=self.weight)
        pt = torch.exp(-bce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * bce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss
