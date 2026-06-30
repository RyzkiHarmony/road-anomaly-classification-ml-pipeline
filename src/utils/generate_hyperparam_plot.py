import os
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def load_results(path):
    if not os.path.exists(path):
        print(f"Warning: {path} not found.")
        return {}
    with open(path, 'r') as f:
        return json.load(f)

def plot_sensitivity(ax, param_values, loss_values, f1_values, xlabel, title, is_log_x=False):
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    color_loss = 'tab:red'
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel('Avg CV Loss', color=color_loss, fontsize=12)
    
    # Extract data handling potential missing string keys
    if isinstance(param_values[0], str):
        # Numeric keys might have been stringified in JSON
        try:
            x_vals = [float(p) for p in param_values]
        except ValueError:
            x_vals = param_values
    else:
        x_vals = param_values
        
    line1 = ax.plot(x_vals, loss_values, color=color_loss, marker='o', linewidth=2, label='CV Loss')
    ax.tick_params(axis='y', labelcolor=color_loss)
    
    if is_log_x:
        ax.set_xscale('log')
        
    ax2 = ax.twinx()
    color_f1 = 'tab:blue'
    ax2.set_ylabel('Macro F1-Score', color=color_f1, fontsize=12)
    line2 = ax2.plot(x_vals, f1_values, color=color_f1, marker='s', linewidth=2, linestyle='--', label='Macro F1')
    ax2.tick_params(axis='y', labelcolor=color_f1)
    ax2.set_ylim(0, 1.0)
    
    # Combined legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, loc='best')

def main():
    base_dir = os.path.join("src", "cnn_model", "analysis")
    lr_res = load_results(os.path.join(base_dir, "lr_tune_results.json"))
    ep_res = load_results(os.path.join(base_dir, "epoch_tune_results.json"))
    bs_res = load_results(os.path.join(base_dir, "batch_sz_tune_results.json"))
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Analisis Sensitivitas Hyperparameter 1D-CNN", fontsize=18, fontweight='bold', y=1.05)
    
    # Learning Rate
    if lr_res and isinstance(lr_res, list):
        lrs = [item["lr"] for item in lr_res]
        loss = [item["val_loss"] for item in lr_res]
        f1 = [item["val_macro_f1"] for item in lr_res]
        # Sort together
        data = sorted(zip(lrs, loss, f1), key=lambda x: float(x[0]))
        lrs, loss, f1 = zip(*data)
        plot_sensitivity(axes[0], lrs, loss, f1, "Learning Rate", "Variasi Learning Rate", is_log_x=True)
        
    # Epoch
    if ep_res and isinstance(ep_res, list):
        eps = [item["epochs"] for item in ep_res]
        loss = [item["val_loss"] for item in ep_res]
        f1 = [item["val_macro_f1"] for item in ep_res]
        data = sorted(zip(eps, loss, f1), key=lambda x: int(x[0]))
        eps, loss, f1 = zip(*data)
        plot_sensitivity(axes[1], eps, loss, f1, "Epochs", "Variasi Epoch")
        
    # Batch Size
    if bs_res and isinstance(bs_res, list):
        bss = [item["batch_size"] for item in bs_res]
        loss = [item["val_loss"] for item in bs_res]
        f1 = [item["val_macro_f1"] for item in bs_res]
        data = sorted(zip(bss, loss, f1), key=lambda x: int(x[0]))
        bss, loss, f1 = zip(*data)
        plot_sensitivity(axes[2], bss, loss, f1, "Batch Size", "Variasi Batch Size")
        
    plt.tight_layout()
    out_dir = os.path.join("evaluation", "reports", "hyperparameters")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "hyperparam_sensitivity.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Grafik sensitivitas berhasil disimpan ke {out_path}")

if __name__ == "__main__":
    main()
