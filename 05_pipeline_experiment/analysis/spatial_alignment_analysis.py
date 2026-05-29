# spatial_alignment_analysis.py
import os
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sensor_fusion import apply_sensor_fusion
from peak_detection import detect_peaks
from feature_extraction import extract_event_shape_features
from config import OUT_FOLDER, get_logger

logger = get_logger(__name__)

def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371000.0  # Earth radius in meters
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)
    a = np.sin(delta_phi/2.0)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda/2.0)**2
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return R * c

def run_spatial_alignment():
    csv_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "new-data", "csv")
    legacy_csv_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "csv")
    
    # Ambil file trip 25 Mei (Legacy - LPF Software) dan 26 Mei (New - Android Native)
    files_25 = glob.glob(os.path.join(csv_dir, "*2026-05-25*.csv"))
    if not files_25:
        # Fallback to legacy path for May 25 file if not present in new-data
        files_25 = glob.glob(os.path.join(legacy_csv_dir, "*2026-05-25*.csv"))
        
    files_26 = glob.glob(os.path.join(csv_dir, "*2026-05-26*.csv"))
    
    if not files_25 or not files_26:
        logger.error("Missing trip files from May 25 or May 26 to perform spatial validation.")
        return
        
    file_25 = files_25[0]
    file_26 = files_26[0]
    
    logger.info(f"Comparing Spatial Alignment between:")
    logger.info(f"  • Trip 25 Mei (Legacy Software Path): {os.path.basename(file_25)}")
    logger.info(f"  • Trip 26 Mei (Native Hardware Path): {os.path.basename(file_26)}")
    
    # Load and apply sensor fusion
    logger.info("Applying sensor fusion on May 25 trip (falling back to legacy LPF)...")
    df_25_raw = pd.read_csv(file_25)
    df_25_fused = apply_sensor_fusion(df_25_raw)
    
    logger.info("Applying sensor fusion on May 26 trip (using native Android fusion)...")
    df_26_raw = pd.read_csv(file_26)
    df_26_fused = apply_sensor_fusion(df_26_raw)
    
    # Peak / Candidate Detection
    logger.info("Detecting road anomaly peaks on both trips...")
    peaks_25, _, _, _ = detect_peaks(df_25_fused)
    peaks_26, _, _, _ = detect_peaks(df_26_fused)
    
    logger.info(f"Detected peaks: 25 Mei = {len(peaks_25)} events, 26 Mei = {len(peaks_26)} events.")
    
    if peaks_25.empty or peaks_26.empty:
        logger.error("No peaks detected in one of the trips. Cannot match events.")
        return
        
    # Match peaks spatially using GPS (threshold: 10 meter)
    MATCH_DIST_THRESHOLD_M = 10.0
    matched_pairs = []
    
    for i, row26 in peaks_26.iterrows():
        lat26, lon26 = row26["lat"], row26["lon"]
        
        # Calculate distances to all peaks in 25 Mei trip
        dists = haversine_distance(lat26, lon26, peaks_25["lat"].values, peaks_25["lon"].values)
        min_idx = np.argmin(dists)
        min_dist = dists[min_idx]
        
        if min_dist <= MATCH_DIST_THRESHOLD_M:
            matched_pairs.append({
                "peak_26_idx": i,
                "peak_25_idx": min_idx,
                "distance_m": min_dist,
                "lat": lat26,
                "lon": lon26,
                "time_26": row26["time_s"],
                "time_25": peaks_25.iloc[min_idx]["time_s"]
            })
            
    logger.info(f"Successfully matched {len(matched_pairs)} physical road anomalies between both trips!")
    
    if len(matched_pairs) < 3:
        logger.warning("Insufficient matched events to compute statistically robust correlations.")
        return
        
    # Extract event features for matched pairs
    comparison_records = []
    
    for pair in matched_pairs:
        # Extract features for May 25 (Software)
        feats_25 = extract_event_shape_features(df_25_fused, pair["time_25"])
        
        # Extract features for May 26 (Native Android Hardware)
        feats_26 = extract_event_shape_features(df_26_fused, pair["time_26"])
        
        if feats_25 and feats_26:
            comparison_records.append({
                "distance_m": pair["distance_m"],
                # Peak to Peak (G)
                "p2p_25": feats_25["peak_to_peak"] / 9.80665,
                "p2p_26": feats_26["peak_to_peak"] / 9.80665,
                # Jerk (G/s)
                "jerk_25": feats_25["max_jerk"] / 9.80665,
                "jerk_26": feats_26["max_jerk"] / 9.80665,
                # Vertical Energy
                "energy_25": feats_25["vertical_energy"],
                "energy_26": feats_26["vertical_energy"],
                # Speed Normalized Peak-to-Peak
                "sn_p2p_25": feats_25["speed_normalized_p2p"],
                "sn_p2p_26": feats_26["speed_normalized_p2p"],
            })
            
    df_feats = pd.DataFrame(comparison_records)
    
    # Calculate feature correlations
    p2p_corr, _ = pearsonr(df_feats["p2p_25"].values, df_feats["p2p_26"].values)
    jerk_corr, _ = pearsonr(df_feats["jerk_25"].values, df_feats["jerk_26"].values)
    energy_corr, _ = pearsonr(df_feats["energy_25"].values, df_feats["energy_26"].values)
    sn_p2p_corr, _ = pearsonr(df_feats["sn_p2p_25"].values, df_feats["sn_p2p_26"].values)
    
    print("\n" + "="*60)
    print(f"{'CROSS-TRIP SPATIAL EVENT ALIGNMENT REPORT':^60}")
    print("="*60)
    print(f"Matched Anomalies    : {len(df_feats)} physical hazards")
    print(f"P2P Amplitude Corr   : {p2p_corr:.4f}  (Target: >0.80)")
    print(f"Max Jerk Corr        : {jerk_corr:.4f}")
    print(f"Vertical Energy Corr : {energy_corr:.4f}")
    print(f"Speed Norm P2P Corr  : {sn_p2p_corr:.4f}")
    print("-" * 60)
    
    avg_corr = np.mean([p2p_corr, jerk_corr, energy_corr, sn_p2p_corr])
    print(f"Average Feature Corr : {avg_corr:.4f}")
    
    if avg_corr >= 0.75:
        print("STATUS: SUCCESS (Fitur anomali fisik terbukti konsisten lintas trip dan sensor!)")
    else:
        print("STATUS: WARNING (Ada perbedaan pembacaan anomali fisik akibat faktor getaran/kecepatan!)")
    print("="*60 + "\n")
    
    # Create beautiful scatter validation plot
    plt.figure(figsize=(12, 10))
    
    # 1. Peak to Peak Scatter
    plt.subplot(2, 2, 1)
    plt.scatter(df_feats["p2p_25"], df_feats["p2p_26"], color="#1f77b4", alpha=0.8, edgecolors='k')
    # draw diagonal 1:1 line
    lims = [0, max(plt.xlim()[1], plt.ylim()[1])]
    plt.plot(lims, lims, 'k--', alpha=0.5)
    plt.title(f"Peak-to-Peak Amplitude (G)\nCorrelation: {p2p_corr:.4f}")
    plt.xlabel("Legacy Soft Fusion (25 Mei)")
    plt.ylabel("Native Android Fusion (26 Mei)")
    plt.grid(True, linestyle=":", alpha=0.6)
    
    # 2. Max Jerk Scatter
    plt.subplot(2, 2, 2)
    plt.scatter(df_feats["jerk_25"], df_feats["jerk_26"], color="#ff7f0e", alpha=0.8, edgecolors='k')
    lims = [0, max(plt.xlim()[1], plt.ylim()[1])]
    plt.plot(lims, lims, 'k--', alpha=0.5)
    plt.title(f"Maximum Jerk (G/s)\nCorrelation: {jerk_corr:.4f}")
    plt.xlabel("Legacy Soft Fusion (25 Mei)")
    plt.ylabel("Native Android Fusion (26 Mei)")
    plt.grid(True, linestyle=":", alpha=0.6)
    
    # 3. Vertical Energy Scatter
    plt.subplot(2, 2, 3)
    plt.scatter(df_feats["energy_25"], df_feats["energy_26"], color="#2ca02c", alpha=0.8, edgecolors='k')
    lims = [0, max(plt.xlim()[1], plt.ylim()[1])]
    plt.plot(lims, lims, 'k--', alpha=0.5)
    plt.title(f"Vertical Energy\nCorrelation: {energy_corr:.4f}")
    plt.xlabel("Legacy Soft Fusion (25 Mei)")
    plt.ylabel("Native Android Fusion (26 Mei)")
    plt.grid(True, linestyle=":", alpha=0.6)
    
    # 4. Speed Normalized P2P
    plt.subplot(2, 2, 4)
    plt.scatter(df_feats["sn_p2p_25"], df_feats["sn_p2p_26"], color="#d62728", alpha=0.8, edgecolors='k')
    lims = [0, max(plt.xlim()[1], plt.ylim()[1])]
    plt.plot(lims, lims, 'k--', alpha=0.5)
    plt.title(f"Speed-Normalized P2P\nCorrelation: {sn_p2p_corr:.4f}")
    plt.xlabel("Legacy Soft Fusion (25 Mei)")
    plt.ylabel("Native Android Fusion (26 Mei)")
    plt.grid(True, linestyle=":", alpha=0.6)
    
    plt.suptitle("CROSS-TRIP PHYSICAL ANOMALY FEATURE CORRELATION (25 MEI vs 26 MEI)\nValidating Software LPF vs. Native Android Hardware Sensor Fusion", fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    plot_path = os.path.join(OUT_FOLDER, "cross_trip_spatial_alignment.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    
    logger.info(f"Cross-trip validation plot saved at: {plot_path}")

if __name__ == "__main__":
    run_spatial_alignment()
