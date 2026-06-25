import sys
import os
# Ensure parent directory is in path for modules
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(_SCRIPT_DIR))

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Adjust paths to use absolute locations relative to this script
DATA_PATH = os.path.abspath(os.path.join(_SCRIPT_DIR, '..', 'out', 'manual_labeled_windows.csv'))
OUT_DIR = os.path.abspath(os.path.join(_SCRIPT_DIR, '..', 'out', 'plots'))

if not os.path.exists(DATA_PATH):
    print(f"Dataset not found at {DATA_PATH}")
    sys.exit(1)

df = pd.read_csv(DATA_PATH)

# Select key features for correlation
key_features = ['peak_mag', 'peak_vertical_g', 'speed_mean', 'vertical_energy', 'gyro_energy', 'max_jerk', 'peak_to_peak', 'zcr']
available_features = [f for f in key_features if f in df.columns]

plt.figure(figsize=(12, 10))
corr = df[available_features].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Heatmap Korelasi Antar Fitur Utama', fontsize=15, pad=20)
plt.tight_layout()

os.makedirs(OUT_DIR, exist_ok=True)
plot_path = os.path.join(OUT_DIR, 'feature_correlation.png')
plt.savefig(plot_path, dpi=300)
print(f"Saved: {plot_path}")
