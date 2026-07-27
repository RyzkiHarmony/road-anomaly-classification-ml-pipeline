import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import torch

# Tambahkan path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.utils.config import OUT_FOLDER

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
        x = self.X[idx].clone()
        y_val = self.y[idx]
        if self.is_train:
            # 1. Random Crop
            seq_len = 200
            if x.shape[-1] > seq_len:
                max_start_idx = x.shape[-1] - seq_len
                start_idx = np.random.randint(0, max_start_idx + 1) if self.max_jitter > 0 else max_start_idx // 2
                x = x[..., start_idx:start_idx + seq_len]
            elif x.shape[-1] == seq_len:
                pass
            
            # 2. Time Warping
            if self.time_warp_prob > 0 and np.random.rand() < self.time_warp_prob:
                T = x.shape[-1]
                n_knots = 4
                knot_positions = np.linspace(0, T - 1, n_knots + 2)
                knot_offsets = np.random.uniform(-self.time_warp_mag * T, self.time_warp_mag * T, size=n_knots + 2)
                knot_offsets[0] = 0
                knot_offsets[-1] = 0
                orig_indices = np.arange(T, dtype=np.float32)
                warped_indices = np.interp(orig_indices, knot_positions, knot_positions + knot_offsets)
                warped_indices = np.clip(warped_indices, 0, T - 1)
                warped_int = warped_indices.astype(np.int64)
                warped_frac = warped_indices - warped_int
                warped_int_next = np.minimum(warped_int + 1, T - 1)
                warped_frac_t = torch.from_numpy(warped_frac).float().unsqueeze(0)
                x = x[:, warped_int] * (1 - warped_frac_t) + x[:, warped_int_next] * warped_frac_t
            
            # 3. Gaussian Noise
            if self.noise_std > 0 and np.random.rand() < 1.0:
                noise = torch.randn_like(x) * self.noise_std
                x = x + noise
            
            # 4. Magnitude Scaling
            if self.scale_range is not None and np.random.rand() < 1.0:
                lo, hi = self.scale_range
                n_channels = x.shape[0]
                scale = torch.FloatTensor(n_channels, 1).uniform_(lo, hi)
                x = x * scale
                
            # 5. Channel Dropout
            if self.channel_drop_prob > 0 and np.random.rand() < 1.0:
                n_channels = x.shape[0]
                drop_mask = torch.rand(n_channels, 1) > self.channel_drop_prob
                if drop_mask.any():
                    x = x * drop_mask.float()
                    
        return x, y_val

def plot_signal(ax, x, title, is_original=False):
    # Asumsikan X bentuknya [channels, timesteps]
    # Plot Y-axis accelerometer (index 1 untuk vertikal)
    color = 'blue' if is_original else 'green'
    ax.plot(x[1, :].numpy(), color=color, linewidth=1.5, label='Acc-Y (Vertikal)')
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_xlim(0, 200)
    ax.grid(True, linestyle='--', alpha=0.5)
    if is_original:
        ax.legend(loc='upper right')

def main():
    print("Memuat dataset...")
    X_path = os.path.join("data", "processed", "cnn_1d", "cnn_1d_X.npy")
    y_path = os.path.join("data", "processed", "cnn_1d", "cnn_1d_y.npy")
    
    if not os.path.exists(X_path):
        print(f"Error: Dataset {X_path} tidak ditemukan.")
        return
        
    X_full = np.load(X_path)
    y_full = np.load(y_path)
    
    # Cari indeks pertama untuk kelas Pothole
    pothole_indices = np.where(y_full == 'Pothole')[0]
    if len(pothole_indices) == 0:
        print("Tidak ada sampel Pothole!")
        return
        
    idx = pothole_indices[12] # Ambil sampel ke-13 agar mendapatkan bentuk sinyal yang berbeda
    
    # Bungkus dalam tensor untuk dataloader
    X_tensor = torch.tensor(X_full[idx:idx+1], dtype=torch.float32)
    # y doesn't matter for augmentation output, just pass dummy tensor
    y_tensor = torch.tensor([1], dtype=torch.long)
    
    # 1. Original (Hanya Center Crop)
    ds_orig = DynamicJitterDataset(X_tensor, y_tensor, 
                                   max_jitter=0, time_warp_prob=0.0, 
                                   noise_std=0.0, scale_range=(1.0, 1.0),
                                   channel_drop_prob=0.0, is_train=True)
    x_orig, _ = ds_orig[0]
    
    # 2. Random Crop (Position Jitter)
    np.random.seed(42)
    ds_crop = DynamicJitterDataset(X_tensor, y_tensor, 
                                   max_jitter=20, time_warp_prob=0.0, 
                                   noise_std=0.0, scale_range=(1.0, 1.0),
                                   channel_drop_prob=0.0, is_train=True)
    x_crop, _ = ds_crop[0]
    
    # 3. Time Warping
    np.random.seed(45)
    ds_warp = DynamicJitterDataset(X_tensor, y_tensor, 
                                   max_jitter=0, time_warp_prob=1.0, time_warp_mag=0.2,
                                   noise_std=0.0, scale_range=(1.0, 1.0),
                                   channel_drop_prob=0.0, is_train=True)
    x_warp, _ = ds_warp[0]
    
    # 4. Gaussian Noise
    torch.manual_seed(42)
    np.random.seed(42)
    ds_noise = DynamicJitterDataset(X_tensor, y_tensor, 
                                   max_jitter=0, time_warp_prob=0.0, 
                                   noise_std=0.2, scale_range=(1.0, 1.0),
                                   channel_drop_prob=0.0, is_train=True)
    x_noise, _ = ds_noise[0]
    
    # 5. Magnitude Scaling
    torch.manual_seed(12)
    np.random.seed(12)
    ds_scale = DynamicJitterDataset(X_tensor, y_tensor, 
                                   max_jitter=0, time_warp_prob=0.0, 
                                   noise_std=0.0, scale_range=(0.4, 0.5),
                                   channel_drop_prob=0.0, is_train=True)
    x_scale, _ = ds_scale[0]
    
    # Setup Figure
    fig, axes = plt.subplots(5, 1, figsize=(10, 12))
    
    plot_signal(axes[0], x_orig, "1. Sinyal Asli (Original Pothole)", is_original=True)
    plot_signal(axes[1], x_crop, "2. Random Crop (Pergeseran Posisi Spasial)")
    plot_signal(axes[2], x_warp, "3. Time Warping (Deformasi Waktu Non-Linier)")
    plot_signal(axes[3], x_noise, "4. Gaussian Noise Injection (Penambahan Derau)")
    plot_signal(axes[4], x_scale, "5. Magnitude Scaling (Penskalaan Amplitudo)")
    
    plt.suptitle("Visualisasi Teknik Augmentasi Data Sinyal (Kelas Pothole)", fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    out_dir = os.path.join(OUT_FOLDER, "reports", "cnn_1d")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "augmentation_examples.png")
    
    plt.savefig(out_path, dpi=300)
    print(f"Visualisasi augmentasi berhasil disimpan di: {out_path}")

if __name__ == "__main__":
    main()
