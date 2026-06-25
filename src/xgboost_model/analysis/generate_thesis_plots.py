import sys
import os
# Ensure parent directory is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Setup style
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 12})

# Adjust paths to use absolute locations relative to this script
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.abspath(os.path.join(_SCRIPT_DIR, '..', 'out', 'manual_labeled_windows.csv'))
OUT_DIR = os.path.abspath(os.path.join(_SCRIPT_DIR, '..', 'out', 'plots'))
os.makedirs(OUT_DIR, exist_ok=True)

if not os.path.exists(DATA_PATH):
    print("Dataset not found.")
    exit(1)

df = pd.read_csv(DATA_PATH)

# --- 1. Class Distribution ---
plt.figure(figsize=(10, 6))
ax = sns.countplot(data=df, x='label', palette='viridis', hue='label', legend=False)
plt.title('Distribusi Kelas Dataset Road Anomaly', fontsize=15, pad=20)
plt.xlabel('Kategori Anomali', fontsize=12)
plt.ylabel('Jumlah Sampel (Termasuk Augmentasi)', fontsize=12)

# Add counts on top
for p in ax.patches:
    ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', xytext=(0, 10), textcoords='offset points')

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'dataset_distribution.png'), dpi=300)
print(f"Saved: {os.path.join(OUT_DIR, 'dataset_distribution.png')}")

# --- 2. Feature Separability (Vertical Energy) ---
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='label', y='vertical_energy', palette='magma', hue='label', legend=False)
plt.yscale('log') # Use log scale for better visualization of energy
plt.title('Karakteristik Energi Vertikal per Kategori', fontsize=15, pad=20)
plt.xlabel('Kategori Anomali', fontsize=12)
plt.ylabel('Vertical Energy (Log Scale)', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'feature_separability.png'), dpi=300)
print(f"Saved: {os.path.join(OUT_DIR, 'feature_separability.png')}")

# --- 3. Source Distribution (Original vs Augmented) ---
if 'source' in df.columns:
    plt.figure(figsize=(8, 8))
    source_counts = df['source'].apply(lambda x: 'Augmented' if 'augmented' in str(x).lower() else 'Original').value_counts()
    plt.pie(source_counts, labels=source_counts.index, autopct='%1.1f%%', colors=['#66b3ff','#99ff99'], startangle=140)
    plt.title('Komposisi Data Original vs Augmentasi', fontsize=15)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'data_source_composition.png'), dpi=300)
    print(f"Saved: {os.path.join(OUT_DIR, 'data_source_composition.png')}")
