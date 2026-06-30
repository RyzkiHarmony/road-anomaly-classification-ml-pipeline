import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    data_dir = os.path.join("data", "processed", "cnn_1d")
    out_path = os.path.join("evaluation", "reports", "sensor_raw_waveform.png")
    
    x_path = os.path.join(data_dir, "cnn_1d_X.npy")
    y_path = os.path.join(data_dir, "cnn_1d_y.npy")
    
    if not os.path.exists(x_path):
        print(f"File {x_path} tidak ditemukan!")
        return
        
    X = np.load(x_path)
    y = np.load(y_path)
    
    print(f"Loaded dataset: X shape {X.shape}, y shape {y.shape}")
    
    non_event_idx = 'Non-Event'
    pothole_idx = 'Pothole'
    sb_idx = 'Speed Bump'
    
    def get_representative_sample(class_idx, maximize=True, rank_offset=4):
        indices = np.where(y == class_idx)[0]
        if len(indices) == 0: return None
        max_vals = []
        for i in indices:
            max_vals.append(np.max(np.abs(X[i, 0, :])))
        sorted_indices = [x for _, x in sorted(zip(max_vals, indices), reverse=True)]
        if maximize:
            best_i = sorted_indices[min(rank_offset, len(sorted_indices)-1)]
        else:
            best_i = sorted_indices[len(sorted_indices) // 2]
        return X[best_i]
        
    sample_ne = get_representative_sample(non_event_idx, maximize=False)
    sample_p = get_representative_sample(pothole_idx, maximize=True, rank_offset=10)
    sample_sb = get_representative_sample(sb_idx, maximize=True, rank_offset=5)
    
    samples = [sample_p, sample_sb, sample_ne]
    titles = ["Pothole", "Speed Bump", "Non-Event"]
    filenames = ["sensor_raw_waveform_pothole.png", "sensor_raw_waveform_speedbump.png", "sensor_raw_waveform_nonevent.png"]
    
    time_axis = np.linspace(0, 2.0, 200)
    
    os.makedirs(os.path.join("evaluation", "reports"), exist_ok=True)
    
    for sample, title, filename in zip(samples, titles, filenames):
        if sample is None:
            continue
            
        fig, axes = plt.subplots(3, 2, figsize=(14, 10), sharex=True)
        fig.suptitle(f"Raw Sensor Data - {title}", fontsize=16, fontweight='bold')
        
        # Accel (Left Column)
        for row, (ch, ax_name, color) in enumerate([(14, 'X', 'red'), (15, 'Y', 'green'), (16, 'Z', 'blue')]):
            ax = axes[row, 0]
            ax.plot(time_axis, sample[ch, :], color=color, linewidth=1.5)
            ax.set_ylabel(f"Accel {ax_name} (m/s²)", fontsize=10)
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.set_ylim(-60, 60)
            
        # Gyro (Right Column)
        for row, (ch, ax_name, color) in enumerate([(5, 'X', 'red'), (6, 'Y', 'green'), (7, 'Z', 'blue')]):
            ax = axes[row, 1]
            ax.plot(time_axis, sample[ch, :], color=color, linewidth=1.5)
            ax.set_ylabel(f"Gyro {ax_name} (rad/s)", fontsize=10)
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.set_ylim(-15, 15)
            
        axes[0, 0].set_title("Linear Accelerometer", fontsize=12, fontweight='bold')
        axes[0, 1].set_title("Raw Gyroscope", fontsize=12, fontweight='bold')
        
        axes[2, 0].set_xlabel("Waktu (Detik) dalam Window 2.0s", fontsize=12)
        axes[2, 1].set_xlabel("Waktu (Detik) dalam Window 2.0s", fontsize=12)
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        out_dir = os.path.join("evaluation", "reports", "waveforms", "raw_linear")
        os.makedirs(out_dir, exist_ok=True)
        
        out_f = os.path.join(out_dir, filename)
        plt.savefig(out_f, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Visualisasi raw sensor untuk {title} berhasil disimpan ke: {out_f}")

if __name__ == "__main__":
    main()
