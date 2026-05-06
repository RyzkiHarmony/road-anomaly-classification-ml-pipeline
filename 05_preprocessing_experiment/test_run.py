import pandas as pd
import glob
from sensor_fusion import apply_sensor_fusion
import sys

csv_files = glob.glob('data/csv/*.csv')
if not csv_files:
    print("No CSV files found.")
    sys.exit(0)

df = pd.read_csv(csv_files[0])
print(f"Original shape: {df.shape}")

try:
    df_fused = apply_sensor_fusion(df)
    print(f"Processed shape: {df_fused.shape}")
    print(f"Head of processed timestamp: {df_fused['timestamp'].head()}")
    print("SUCCESS")
except Exception as e:
    print(f"ERROR: {e}")
    sys.exit(1)
