import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

def generate_heatmaps():
    out_dir = os.path.join("evaluation", "reports")
    os.makedirs(out_dir, exist_ok=True)
    
    # -------------------------------------------------------------
    # 1. XGBoost All Engineered Features Correlation Heatmap
    # -------------------------------------------------------------
    xgb_path = os.path.join("data", "processed", "xgboost", "xgboost_labeled_windows.csv")
    if os.path.exists(xgb_path):
        df_xgb = pd.read_csv(xgb_path)
        meta_cols = ["event_id", "time_s", "lat", "lon", "level", "suggestion_confidence", "speed_factor", "score", "label", "trip_id", "target"]
        
        # Select all numerical engineered feature columns
        all_features = [c for c in df_xgb.select_dtypes(include=["float64", "int64"]).columns if c not in meta_cols]
        print(f"[INFO] Found {len(all_features)} engineered features for XGBoost correlation heatmap.")
        
        corr_xgb = df_xgb[all_features].corr()
        
        # Save full correlation matrix to CSV
        corr_csv_path = os.path.join(out_dir, "xgboost_all_features_correlation_matrix.csv")
        corr_xgb.to_csv(corr_csv_path)
        print(f"[OK] Saved full correlation matrix CSV: {corr_csv_path}")

        # Plot full heatmap
        plt.figure(figsize=(26, 22))
        mask_xgb = np.triu(np.ones_like(corr_xgb, dtype=bool))
        sns.heatmap(
            corr_xgb, mask=mask_xgb, annot=True, fmt=".2f", cmap="coolwarm",
            vmin=-1.0, vmax=1.0, square=True, linewidths=0.3, annot_kws={"size": 6},
            cbar_kws={"shrink": 0.7}
        )
        plt.title(f"XGBoost Full Feature Correlation Heatmap ({len(all_features)} Engineered Features)", fontsize=18, pad=25, fontweight="bold")
        plt.xticks(rotation=90, ha="right", fontsize=8)
        plt.yticks(fontsize=8)
        plt.tight_layout()
        
        xgb_plot_path = os.path.join(out_dir, "heatmap_xgboost_all_features.png")
        plt.savefig(xgb_plot_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"[OK] Saved XGBoost All Features Heatmap: {xgb_plot_path}")

        # -------------------------------------------------------------
        # Feature Selection (|r| >= 0.5)
        # -------------------------------------------------------------
        if "target" in df_xgb.columns or "label" in df_xgb.columns:
            target_series = df_xgb["target"] if "target" in df_xgb.columns else (df_xgb["label"] != "Non-Event").astype(int)
            target_corr = df_xgb[all_features].apply(lambda col: col.corr(target_series)).abs().sort_values(ascending=False)
            
            # Select features with target correlation >= 0.5 (or top strong features >= 0.35/0.5)
            selected_r05 = target_corr[target_corr >= 0.35].index.tolist()
            
            # Save feature selection summary CSV
            sel_df = pd.DataFrame({
                "feature": target_corr.index,
                "abs_correlation": target_corr.values,
                "selected_r05_threshold": target_corr.values >= 0.5,
                "selected_r035_threshold": target_corr.values >= 0.35
            })
            sel_csv_path = os.path.join(out_dir, "feature_selection_r05_summary.csv")
            sel_df.to_csv(sel_csv_path, index=False)
            print(f"[OK] Saved Feature Selection Summary CSV: {sel_csv_path}")

            # Plot Heatmap of Selected Features (|r| >= 0.35 / 0.5)
            if len(selected_r05) > 1:
                corr_sel = df_xgb[selected_r05].corr()
                plt.figure(figsize=(12, 10))
                mask_sel = np.triu(np.ones_like(corr_sel, dtype=bool))
                sns.heatmap(
                    corr_sel, mask=mask_sel, annot=True, fmt=".2f", cmap="coolwarm",
                    vmin=-1.0, vmax=1.0, square=True, linewidths=0.5, annot_kws={"size": 9},
                    cbar_kws={"shrink": 0.8}
                )
                plt.title("XGBoost Selected Features Correlation Heatmap (|r| >= 0.35 Target Correlation)", fontsize=15, pad=20, fontweight="bold")
                plt.xticks(rotation=45, ha="right", fontsize=9)
                plt.yticks(fontsize=9)
                plt.tight_layout()
                
                sel_plot_path = os.path.join(out_dir, "heatmap_xgboost_selected_features_r05.png")
                plt.savefig(sel_plot_path, dpi=300, bbox_inches="tight")
                plt.close()
                print(f"[OK] Saved Selected Features Heatmap: {sel_plot_path}")
    else:
        print(f"[WARN] File not found: {xgb_path}")

    # -------------------------------------------------------------
    # 2. 1D-CNN Input Channel Correlation Heatmap
    # -------------------------------------------------------------
    cnn_path = os.path.join("data", "processed", "cnn_1d", "cnn_1d_X.npy")
    if os.path.exists(cnn_path):
        X_cnn = np.load(cnn_path)  # (N, 7, 230)
        channel_names = ["speed", "ax", "ay", "az", "gx", "gy", "gz"]
        
        # Calculate RMS energy per channel per window
        channel_rms = np.sqrt(np.mean(X_cnn**2, axis=2))
        df_cnn = pd.DataFrame(channel_rms, columns=channel_names)
        corr_cnn = df_cnn.corr()
        
        plt.figure(figsize=(9, 8))
        mask_cnn = np.triu(np.ones_like(corr_cnn, dtype=bool))
        sns.heatmap(
            corr_cnn, mask=mask_cnn, annot=True, fmt=".2f", cmap="viridis",
            vmin=-1.0, vmax=1.0, square=True, linewidths=0.8, cbar_kws={"shrink": 0.8}
        )
        plt.title("1D-CNN Input Channel Correlation Heatmap (RMS Signal Energy)", fontsize=14, pad=18, fontweight="bold")
        plt.xticks(fontsize=11)
        plt.yticks(fontsize=11)
        plt.tight_layout()
        
        cnn_plot_path = os.path.join(out_dir, "heatmap_1dcnn_channels.png")
        plt.savefig(cnn_plot_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"[OK] Saved 1D-CNN Channel Heatmap: {cnn_plot_path}")
    else:
        print(f"[WARN] File not found: {cnn_path}")

if __name__ == "__main__":
    generate_heatmaps()
