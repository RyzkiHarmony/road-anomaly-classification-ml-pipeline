import os
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    data_dir = os.path.join("data", "processed", "cnn_1d")
    out_path = os.path.join("evaluation", "reports", "waveforms", "engineered", "sensor_waveform_comparison.png")
    
    x_path = os.path.join(data_dir, "cnn_1d_X.npy")
    y_path = os.path.join(data_dir, "cnn_1d_y.npy")
    
    if not os.path.exists(x_path):
        print(f"File {x_path} tidak ditemukan!")
        return
        
    X = np.load(x_path)
    y = np.load(y_path)
    
    print(f"Loaded dataset: X shape {X.shape}, y shape {y.shape}")
    
    # Indices for the classes mapping in train.py
    non_event_idx = 'Non-Event'
    pothole_idx = 'Pothole'
    sb_idx = 'Speed Bump'
    
    # Find samples for each class
    # To get a "good" representative sample, we can pick the one with the highest max absolute a_vertical 
    # for Pothole and Speed Bump, and a random/average one for Non-Event.
    
    def get_representative_sample(class_idx, maximize=True, rank_offset=4):
        indices = np.where(y == class_idx)[0]
        if len(indices) == 0:
            return None
            
        # Hitung max absolute a_vertical untuk semua sampel di kelas ini
        max_vals = []
        for i in indices:
            max_vals.append(np.max(np.abs(X[i, 0, :])))
            
        # Urutkan berdasarkan max_vals
        sorted_indices = [x for _, x in sorted(zip(max_vals, indices), reverse=True)]
        
        if maximize:
            # Ambil yang ke-(rank_offset + 1) terbesar untuk menghindari outlier paling ekstrim
            best_i = sorted_indices[min(rank_offset, len(sorted_indices)-1)]
        else:
            # Untuk non-event, ambil nilai median (tengah)
            best_i = sorted_indices[len(sorted_indices) // 2]
            
        return X[best_i]
        
    sample_ne = get_representative_sample(non_event_idx, maximize=False)
    sample_p = get_representative_sample(pothole_idx, maximize=True, rank_offset=10) # Ambil pothole terbesar ke-11
    sample_sb = get_representative_sample(sb_idx, maximize=True, rank_offset=5) # Ambil speed bump terbesar ke-6
    
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    
    time_axis = np.linspace(0, 2.0, 200) # 2 seconds, 200 samples
    
    # Plot configurations
    samples = [sample_p, sample_sb, sample_ne]
    titles = ["Pothole (Lubang)", "Speed Bump (Polisi Tidur)", "Non-Event (Jalan Normal)"]
    colors = [('red', 'orange', 'purple'), ('blue', 'cyan', 'green'), ('gray', 'silver', 'black')]
    
    for i, (sample, title) in enumerate(zip(samples, titles)):
        ax = axes[i]
        if sample is not None:
            # Smooth magnitude_deviation (channel 17) for visualization because it lacks the 6Hz LPF applied to a_vertical
            window_len = 5
            mag_smooth = np.convolve(sample[17, :], np.ones(window_len)/window_len, mode='same')
            
            # a_vertical is channel 0, a_horizontal is 1, magnitude_deviation is 17
            ax.plot(time_axis, sample[0, :], label='a_vertical (Guncangan Vertikal)', color=colors[i][0], linewidth=2)
            ax.plot(time_axis, sample[1, :], label='a_horizontal (Guncangan Horizontal)', color=colors[i][1], linewidth=1.5, alpha=0.8)
            ax.plot(time_axis, mag_smooth, label='magnitude_deviation (Deviasi dari 1G)', color=colors[i][2], linewidth=1.5, linestyle='--')
            
            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.set_ylabel("Akselerasi (m/s²)", fontsize=12)
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.legend(loc='upper right')
            
            # Set Y limits to be the same across all plots for fair comparison
            ax.set_ylim(-100, 100)
            
    axes[-1].set_xlabel("Waktu (Detik) dalam Window 2.0s", fontsize=12)
    plt.tight_layout()
    
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Visualisasi gelombang sensor berhasil disimpan ke: {out_path}")

if __name__ == "__main__":
    main()
