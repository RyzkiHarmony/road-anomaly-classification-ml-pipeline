import re
import matplotlib.pyplot as plt
import argparse
import os

def plot_loss_from_log(log_path):
    if not os.path.exists(log_path):
        print(f"Error: Log file not found at {log_path}")
        return

    with open(log_path, 'r') as f:
        lines = f.readlines()

    folds_data = {}
    current_fold = None

    # Regex patterns
    fold_pattern = re.compile(r'===\s+Fold\s+(\d+)\s+===')
    epoch_pattern = re.compile(r'\[Epoch\s+\d+/\d+\]\s+Train Loss:\s+([\d.]+)\s+\|\s+Val Loss:\s+([\d.]+)(?:\s+\|\s+Val Macro F1:\s+([\d.]+))?')

    for line in lines:
        # Check for Fold indicator
        fold_match = fold_pattern.search(line)
        if fold_match:
            current_fold = int(fold_match.group(1))
            folds_data[current_fold] = {'train_loss': [], 'val_loss': [], 'val_f1': []}
            continue
        
        # Extract epoch loss if inside a fold
        if current_fold is not None:
            epoch_match = epoch_pattern.search(line)
            if epoch_match:
                train_loss = float(epoch_match.group(1))
                val_loss = float(epoch_match.group(2))
                folds_data[current_fold]['train_loss'].append(train_loss)
                folds_data[current_fold]['val_loss'].append(val_loss)
                
                if epoch_match.group(3):
                    val_f1 = float(epoch_match.group(3))
                    folds_data[current_fold]['val_f1'].append(val_f1)

    if not folds_data:
        print("No fold data found in the log file.")
        return

    num_folds = len(folds_data)
    cols = 2
    rows = (num_folds + cols - 1) // cols
    
    plt.figure(figsize=(15, 5 * rows))
    
    for i, (fold, data) in enumerate(folds_data.items(), 1):
        ax1 = plt.subplot(rows, cols, i)
        epochs = range(1, len(data['train_loss']) + 1)
        
        ax1.plot(epochs, data['train_loss'], label='Train Loss', marker='o', markersize=3, color='blue')
        ax1.plot(epochs, data['val_loss'], label='Val Loss', marker='o', markersize=3, color='orange')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss', color='black')
        ax1.legend(loc='upper left')
        ax1.grid(True, linestyle='--', alpha=0.7)
        
        if data['val_f1']:
            ax2 = ax1.twinx()
            ax2.plot(epochs, data['val_f1'], label='Val Macro F1', marker='s', markersize=3, color='green')
            ax2.set_ylabel('Macro F1-Score', color='green')
            ax2.tick_params(axis='y', labelcolor='green')
            ax2.legend(loc='upper right')
            
        plt.title(f'Fold {fold} Training Metrics')
        
    plt.tight_layout()
    output_img = log_path.replace('.log', '_loss_curve.png')
    plt.savefig(output_img)
    print(f"Plot saved to: {output_img}")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot loss curves from a training log file.")
    parser.add_argument(
        "--log_file", 
        type=str, 
        default=r"d:\VSCode Data\Road Detection - Project Skripsi\ml_pipelines\log\1dcnn_17_tuned.log", 
        help="Path to the log file"
    )
    args = parser.parse_args()
    
    plot_loss_from_log(args.log_file)
