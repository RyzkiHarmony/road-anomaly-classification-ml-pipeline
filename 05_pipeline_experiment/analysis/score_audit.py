import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    # Resolve absolute paths relative to script location
    _SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.abspath(os.path.join(_SCRIPT_DIR, '..', 'out', 'candidates_events.csv'))
    
    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 1. Accel
    if 'peak_vertical_g' in df.columns:
        axes[0].hist(df['peak_vertical_g'], bins=30, color='blue', alpha=0.7)
        axes[0].set_title('Vertical Accel (G)')
        axes[0].set_xlabel('G-Force')
        axes[0].set_ylabel('Count')

    # 2. Jerk
    if 'max_jerk' in df.columns:
        axes[1].hist(df['max_jerk'], bins=30, color='red', alpha=0.7)
        axes[1].set_title('Max Jerk (m/s^3)')
        axes[1].set_xlabel('Jerk')

    # 3. Composite Score
    if 'score' in df.columns:
        axes[2].hist(df['score'], bins=30, color='green', alpha=0.7)
        axes[2].set_title('Composite Score')
        axes[2].set_xlabel('Score')

    plt.tight_layout()
    out_file = os.path.abspath(os.path.join(_SCRIPT_DIR, '..', 'out', 'score_distribution.png'))
    plt.savefig(out_file)
    print(f"Saved to {out_file}")

if __name__ == '__main__':
    main()
