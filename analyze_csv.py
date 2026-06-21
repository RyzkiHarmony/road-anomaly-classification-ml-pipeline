import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load CSV
csv_path = r"d:\VSCode Data\Road Detection - Project Skripsi\android-app\RoadAnomalyDetector\app\src\main\java\com\pemalang\roaddamage\temp\RoadDamage_2026-06-11_21-21-57.csv"
df = pd.read_csv(csv_path)

print(f"Loaded {len(df)} rows.")

# Calculate time in seconds from start
start_time = df['timestamp'].iloc[0]
df['time_sec'] = (df['timestamp'] - start_time) / 1000.0

# 1. Normalize Gravity
g_mag = np.sqrt(df['grav_x']**2 + df['grav_y']**2 + df['grav_z']**2).replace(0, 1)
gx_norm = df['grav_x'] / g_mag
gy_norm = df['grav_y'] / g_mag
gz_norm = df['grav_z'] / g_mag

# 2. Project linear acceleration onto gravity vector (a_vertical)
df['a_vertical'] = (df['lin_ax'] * gx_norm + 
                    df['lin_ay'] * gy_norm + 
                    df['lin_az'] * gz_norm)

potholes_50 = df[df['prob_pothole'] > 0.51]
bumps_50 = df[df['prob_speedbump'] > 0.51]

print(f"Rows with prob_pothole > 0.51: {len(potholes_50)}")
print(f"Rows with prob_speedbump > 0.51: {len(bumps_50)}")

# Plot the time series
plt.figure(figsize=(15, 8))

# Subplot 1: Vertical Acceleration
plt.subplot(3, 1, 1)
plt.plot(df['time_sec'], df['a_vertical'], label='Vertical Acceleration (m/s^2)', color='blue', alpha=0.7)
plt.axhline(0, color='black', linewidth=1)
plt.ylabel('m/s^2')
plt.title('Sensor Data & Model Predictions')
plt.legend(loc='upper right')
plt.grid(True)

# Subplot 2: Probabilities
plt.subplot(3, 1, 2)
plt.plot(df['time_sec'], df['prob_pothole'], label='Prob Pothole', color='red')
plt.plot(df['time_sec'], df['prob_speedbump'], label='Prob Speedbump', color='orange')
plt.axhline(0.51, color='green', linestyle='--', label='Threshold (0.51)')
plt.ylim(0, 1.1)
plt.ylabel('Probability')
plt.legend(loc='upper right')
plt.grid(True)

# Subplot 3: Speed
plt.subplot(3, 1, 3)
plt.plot(df['time_sec'], df['speed'], label='Speed (m/s)', color='purple')
plt.ylabel('Speed (m/s)')
plt.xlabel('Time (seconds)')
plt.legend(loc='upper right')
plt.grid(True)

plt.tight_layout()
plt.savefig(r'C:\Users\ACER\.gemini\antigravity-ide\brain\d3b0ce3b-52e2-4e09-8a2f-c45147a82580\plot.png')
print("Plot saved to plot.png")
