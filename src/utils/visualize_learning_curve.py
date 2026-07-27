import re
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

def parse_log(log_path):
    epochs = 50
    folds = 4
    train_loss = np.zeros((folds, epochs))
    val_loss = np.zeros((folds, epochs))
    val_f1 = np.zeros((folds, epochs))
    
    current_fold = -1
    with open(log_path, 'r') as f:
        for line in f:
            if "=== Fold" in line:
                current_fold += 1
            match = re.search(r'\[Epoch (\d+)/\d+\] Train Loss: ([\d.]+) \| Val Loss: ([\d.]+) \| Val Macro F1: ([\d.]+)', line)
            if match and current_fold >= 0 and current_fold < folds:
                ep = int(match.group(1)) - 1
                train_loss[current_fold, ep] = float(match.group(2))
                val_loss[current_fold, ep] = float(match.group(3))
                val_f1[current_fold, ep] = float(match.group(4))
                
def main():
    log_path = 'log/1dcnn_20_baseline.log'
    # We need to rewrite parse_log to return individual folds data, not average.
    epochs = 50
    folds = 4
    train_loss = np.zeros((folds, epochs))
    val_loss = np.zeros((folds, epochs))
    val_f1 = np.zeros((folds, epochs))
    
    current_fold = -1
    with open(log_path, 'r') as f:
        for line in f:
            if "=== Fold" in line:
                current_fold += 1
            match = re.search(r'\[Epoch (\d+)/\d+\] Train Loss: ([\d.]+) \| Val Loss: ([\d.]+) \| Val Macro F1: ([\d.]+)', line)
            if match and current_fold >= 0 and current_fold < folds:
                ep = int(match.group(1)) - 1
                train_loss[current_fold, ep] = float(match.group(2))
                val_loss[current_fold, ep] = float(match.group(3))
                val_f1[current_fold, ep] = float(match.group(4))
    
    epoch_arr = np.arange(1, 51)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.flatten()
    
    for i in range(folds):
        ax1 = axes[i]
        
        ax1.set_title(f'Fold {i+1} Training Metrics', fontsize=12)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss', color='black')
        
        # Plot Loss
        line1 = ax1.plot(epoch_arr, train_loss[i], color='blue', marker='.', label='Train Loss')
        line2 = ax1.plot(epoch_arr, val_loss[i], color='orange', marker='.', label='Val Loss')
        ax1.tick_params(axis='y', labelcolor='black')
        ax1.grid(True, linestyle='--', alpha=0.7)
        
        # Twin axes for F1
        ax2 = ax1.twinx()
        ax2.set_ylabel('Macro F1-Score', color='green')
        line3 = ax2.plot(epoch_arr, val_f1[i], color='green', marker='s', markersize=3, label='Val Macro F1')
        ax2.tick_params(axis='y', labelcolor='green')
        
        # Combine legends
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='upper left')
        
        lines_2 = line3
        labels_2 = [l.get_label() for l in lines_2]
        ax2.legend(lines_2, labels_2, loc='upper right')
        
    plt.tight_layout()
    
    out_dir = 'data/processed/shared/reports/cnn_1d'
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'baseline_learning_curve.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print("Berhasil disimpan di", out_path)

if __name__ == '__main__':
    main()
