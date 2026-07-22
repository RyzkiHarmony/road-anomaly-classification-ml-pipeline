import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def main():
    # Load data
    try:
        X = np.load('data/processed/cnn_1d/cnn_1d_X.npy')
        y = np.load('data/processed/cnn_1d/cnn_1d_y.npy', allow_pickle=True)
    except FileNotFoundError:
        print("Data numpy cnn_1d belum ditemukan, pastikan build_cnn_data.py sudah dijalankan.")
        return
    
    # Target Hz and duration
    target_hz = 100
    # X shape is (N, C, T) -> C: speed, ax, ay, az, gx, gy, gz
    T = X.shape[2]
    time_axis = np.linspace(-T/(2*target_hz), T/(2*target_hz), T)
    
    classes = ['Non-Event', 'Pothole', 'Speed Bump']
    
    # Set modern light mode theme using seaborn
    sns.set_theme(style="whitegrid")
    
    # Custom palette
    # ax/gx: Red-ish (#E63946)
    # ay/gy: Teal (#2A9D8F)
    # az/gz: Orange (#F4A261)
    colors = ['#E63946', '#2A9D8F', '#F4A261']

    fig, axes = plt.subplots(3, 2, figsize=(14, 10), sharex=True)
    fig.suptitle('Visualisasi Segmentasi "Event-Centered Window" (2.3 Detik)', fontweight='bold', fontsize=18, color='#333333')
    
    # Pilih sampel acak
    np.random.seed(42)
    
    for i, cls in enumerate(classes):
        idx = np.where(y == cls)[0]
        if len(idx) == 0:
            continue
        
        # Ambil sampel acak dari kelas ini
        sample_idx = np.random.choice(idx)
        sample = X[sample_idx]
        
        # --- Kolom 1: Accelerometer (ax, ay, az) berada di index 1, 2, 3 ---
        ax_ax = axes[i, 0]
        ax_ax.plot(time_axis, sample[1], label='ax (Forward)', color=colors[0], alpha=0.85, linewidth=2)
        ax_ax.plot(time_axis, sample[2], label='ay (Lateral)', color=colors[1], alpha=0.85, linewidth=2)
        ax_ax.plot(time_axis, sample[3], label='az (Vertical)', color=colors[2], alpha=0.85, linewidth=2)
        
        ax_ax.set_ylabel(f"{cls}\nAccel (m/s²)", fontweight='bold', fontsize=12, color='#444444')
        ax_ax.tick_params(axis='both', colors='#555555')
        
        if i == 0:
            ax_ax.set_title("Sensor Accelerometer", fontsize=14, fontweight='bold', color='#444444', pad=10)
            ax_ax.legend(loc='upper right', frameon=True, shadow=True)
            
        # --- Kolom 2: Gyroscope (gx, gy, gz) berada di index 4, 5, 6 ---
        ax_gy = axes[i, 1]
        ax_gy.plot(time_axis, sample[4], label='gx', color=colors[0], alpha=0.85, linewidth=2)
        ax_gy.plot(time_axis, sample[5], label='gy', color=colors[1], alpha=0.85, linewidth=2)
        ax_gy.plot(time_axis, sample[6], label='gz', color=colors[2], alpha=0.85, linewidth=2)
        
        ax_gy.set_ylabel(f"{cls}\nGyro (rad/s)", fontweight='bold', fontsize=12, color='#444444')
        ax_gy.tick_params(axis='both', colors='#555555')
        
        if i == 0:
            ax_gy.set_title("Sensor Gyroscope", fontsize=14, fontweight='bold', color='#444444', pad=10)
            ax_gy.legend(loc='upper right', frameon=True, shadow=True)
            
        # Garis penanda tengah event (t=0)
        ax_ax.axvline(x=0, color='#888888', linestyle='--', linewidth=1.5, zorder=0)
        ax_gy.axvline(x=0, color='#888888', linestyle='--', linewidth=1.5, zorder=0)
            
    axes[2, 0].set_xlabel("Waktu Relatif terhadap Event (detik)", fontsize=12, fontweight='bold', color='#444444')
    axes[2, 1].set_xlabel("Waktu Relatif terhadap Event (detik)", fontsize=12, fontweight='bold', color='#444444')
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Background color adjustment
    fig.patch.set_facecolor('#F8F9FA')
    for ax in axes.flatten():
        ax.set_facecolor('#FFFFFF')
    
    out_path = 'evaluation/event_centered_segmentation.png'
    os.makedirs('evaluation', exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
    print(f"Saved visualization to {out_path}")

if __name__ == '__main__':
    main()
