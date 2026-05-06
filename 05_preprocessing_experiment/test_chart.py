import pandas as pd
import numpy as np
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from manual_labeling_per_trip import generate_event_chart_b64, CANDIDATE_PATH, TRIP_CSV_MAP
from sensor_fusion import apply_sensor_fusion

df = pd.read_csv(CANDIDATE_PATH)
selected_trip = df['trip_id'].iloc[0]
df_trip = df[df['trip_id'] == selected_trip]
event_time_s = df_trip['time_s'].iloc[0]

raw_csv_path = TRIP_CSV_MAP.get(selected_trip)
print(f"Loading {raw_csv_path}")
df_raw = pd.read_csv(raw_csv_path)

if "magnitude" not in df_raw.columns:
    df_raw["magnitude"] = np.sqrt(df_raw["ax"] ** 2 + df_raw["ay"] ** 2 + df_raw["az"] ** 2)

df_raw = apply_sensor_fusion(df_raw)

print(f"Event time (s): {event_time_s}")
times = df_raw["timestamp"].astype(float) / 1000.0
t_start = event_time_s - 1.5
t_end   = event_time_s + 1.5
seg     = df_raw[(times >= t_start) & (times <= t_end)]
print(f"Segment length: {len(seg)}")

if len(seg) >= 3:
    t_rel = seg["timestamp"].astype(float) / 1000.0 - event_time_s
    ax_val = seg["ax"].astype(float).values
    print("t_rel head:", t_rel.head().values)
    print("ax head:", ax_val[:5])
    print("ax NaNs:", np.isnan(ax_val).sum())
    
    # Try generating chart
    b64 = generate_event_chart_b64(df_raw, event_time_s)
    if b64:
        print("B64 generated length:", len(b64))
        # save as png to check
        import base64
        with open("test_chart.png", "wb") as fh:
            fh.write(base64.b64decode(b64))
        print("Saved to test_chart.png")
    else:
        print("B64 was None")
else:
    print("Segment too short")
