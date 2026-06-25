"""
Comprehensive Data Strategy Ablation Study.
Tests all combinations of: Undersampling, Augmentation, SMOTE.
Uses the SAME CV pipeline as train_model.py for fair comparison.
"""
import sys
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report
from itertools import product

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from config import OUT_FOLDER, BEST_FEATURES, CSV_FOLDER
from build_train_set import augment_anomalies, get_csv_path_for_trip
from sensor_fusion import apply_sensor_fusion
from feature_extraction import extract_event_shape_features

import warnings
warnings.filterwarnings('ignore')

RANDOM_SEED = 42

def build_dataset(do_undersample=True, do_augment=False):
    """Rebuild the dataset with specified strategies (mirrors build_train_set.py logic)."""
    import glob, json
    
    gt_path = os.path.join(OUT_FOLDER, "ground_truth_labels.csv")
    events_path = os.path.join(OUT_FOLDER, "candidates_events.csv")
    
    df_gt = pd.read_csv(gt_path)
    df_events = pd.read_csv(events_path)
    df_labeled = df_events.merge(df_gt[["event_id", "label"]], on="event_id", how="inner")
    
    if df_labeled.empty:
        return pd.DataFrame()

    # Re-extract features
    re_extracted_records = []
    grouped_by_trip = df_labeled.groupby("trip_id")
    for trip_id, group in grouped_by_trip:
        csv_path = get_csv_path_for_trip(trip_id)
        if csv_path:
            try:
                raw_df = pd.read_csv(csv_path)
                raw_df = apply_sensor_fusion(raw_df)
                for _, row in group.iterrows():
                    t_event = row["time_s"]
                    try:
                        feats = extract_event_shape_features(raw_df, t_event)
                        row_dict = row.to_dict()
                        row_dict.update(feats)
                        re_extracted_records.append(row_dict)
                    except:
                        re_extracted_records.append(row.to_dict())
            except:
                for _, row in group.iterrows():
                    re_extracted_records.append(row.to_dict())
        else:
            for _, row in group.iterrows():
                re_extracted_records.append(row.to_dict())
                
    df_labeled = pd.DataFrame(re_extracted_records)

    # Augmentation
    df_augmented = pd.DataFrame()
    if do_augment:
        df_augmented = augment_anomalies(df_labeled)

    # Combine
    dfs = [df_labeled]
    if not df_augmented.empty:
        dfs.append(df_augmented)
    df_final = pd.concat(dfs, ignore_index=True)

    # Clean label noise (same as build_train_set.py)
    is_non_event = df_final['label'] == 'Non-Event'
    is_noisy = is_non_event & (
        (df_final['peak_mag'] > 30.0) | 
        (df_final['crest_factor'] > 4.0) | 
        (df_final['snr_vertical'] > 20.0)
    )
    df_final = df_final[~is_noisy]

    # Undersampling
    if do_undersample:
        n_anomalies = len(df_final[df_final["label"].isin(["Pothole", "Speed Bump"])])
        max_non_events = int(n_anomalies * 1.5)
        df_anomalies = df_final[df_final["label"].isin(["Pothole", "Speed Bump"])]
        df_non_event = df_final[df_final["label"] == "Non-Event"]
        if len(df_non_event) > max_non_events:
            df_non_event = df_non_event.sample(n=max_non_events, random_state=RANDOM_SEED)
        df_final = pd.concat([df_anomalies, df_non_event]).sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    return df_final

def run_cv(df, use_smote=True):
    """Run 5-fold Stratified Group K-Fold CV."""
    feature_cols = [c for c in df.columns if c in BEST_FEATURES]
    df = df.dropna(subset=feature_cols + ['label'])
    
    X = df[feature_cols].values
    y = df['label'].values
    groups = df['trip_id'].values
    
    if 'source' in df.columns:
        source_values = df['source'].fillna('original').values
    else:
        source_values = np.array(['original'] * len(df))

    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    classes = list(le.classes_)
    
    unique_groups = len(np.unique(groups))
    n_folds = min(5, unique_groups) if unique_groups >= 2 else 5
    cv_strategy = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    oof_y_true = []
    oof_y_pred = []
    
    for fold, (train_idx, test_idx) in enumerate(cv_strategy.split(X, y_enc, groups)):
        X_train, y_train = X[train_idx], y_enc[train_idx]
        
        is_original_test = np.array([not str(s).startswith('augmented') for s in source_values[test_idx]])
        clean_test_idx = test_idx[is_original_test]
        X_test, y_test = X[clean_test_idx], y_enc[clean_test_idx]

        missing_classes = set(range(len(classes))) - set(y_train)
        if missing_classes:
            for mc in missing_classes:
                global_idx = np.where(y_enc == mc)[0][0]
                X_train = np.vstack([X_train, X[global_idx]])
                y_train = np.append(y_train, mc)

        if use_smote:
            try:
                smote = SMOTE(random_state=42, k_neighbors=min(5, min(np.bincount(y_train)) - 1))
                if min(np.bincount(y_train)) > 1:
                    X_train, y_train = smote.fit_resample(X_train, y_train)
            except:
                pass

        model = RandomForestClassifier(
            n_estimators=300, max_depth=5, min_samples_leaf=3,
            class_weight='balanced_subsample', random_state=42, n_jobs=-1
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        oof_y_true.extend(y_test)
        oof_y_pred.extend(y_pred)
    
    return np.array(oof_y_true), np.array(oof_y_pred), classes

def main():
    # All combinations: (undersample, augment, smote)
    configs = list(product([True, False], repeat=3))
    
    results = []
    
    print("=" * 80)
    print(f"{'COMPREHENSIVE DATA STRATEGY ABLATION':^80}")
    print(f"{'Testing all 8 combinations of Undersample x Augment x SMOTE':^80}")
    print("=" * 80)
    
    for i, (do_under, do_aug, do_smote) in enumerate(configs):
        tag = f"{'US' if do_under else 'noUS'}_{'AUG' if do_aug else 'noAUG'}_{'SM' if do_smote else 'noSM'}"
        print(f"\n--- [{i+1}/8] {tag} ---")
        print(f"  Undersample: {'YES' if do_under else 'NO'} | Augment: {'YES' if do_aug else 'NO'} | SMOTE: {'YES' if do_smote else 'NO'}")
        
        df = build_dataset(do_undersample=do_under, do_augment=do_aug)
        
        dist = df['label'].value_counts()
        print(f"  Dataset: {len(df)} rows | {dict(dist)}")
        
        oof_true, oof_pred, classes = run_cv(df, use_smote=do_smote)
        report = classification_report(oof_true, oof_pred, target_names=classes, output_dict=True, zero_division=0)
        
        p_idx_name = "Pothole"
        sb_idx_name = "Speed Bump"
        
        row = {
            'Config': tag,
            'Undersample': do_under,
            'Augment': do_aug,
            'SMOTE': do_smote,
            'N_Samples': len(df),
            'P_Precision': report[p_idx_name]['precision'],
            'P_Recall': report[p_idx_name]['recall'],
            'P_F1': report[p_idx_name]['f1-score'],
            'SB_Precision': report[sb_idx_name]['precision'],
            'SB_Recall': report[sb_idx_name]['recall'],
            'SB_F1': report[sb_idx_name]['f1-score'],
            'Macro_F1': report['macro avg']['f1-score'],
            'Accuracy': report['accuracy'],
        }
        results.append(row)
        print(f"  -> Pothole F1: {row['P_F1']:.4f} | Speed Bump F1: {row['SB_F1']:.4f} | Macro F1: {row['Macro_F1']:.4f}")

    # ========== SUMMARY TABLE ==========
    print("\n\n" + "=" * 120)
    print(f"{'FINAL SUMMARY TABLE':^120}")
    print("=" * 120)
    
    header = f"{'Config':<25} {'Samples':>7} {'P_Prec':>7} {'P_Rec':>6} {'P_F1':>6} {'SB_Prec':>8} {'SB_Rec':>7} {'SB_F1':>6} {'Macro':>6} {'Acc':>6}"
    print(header)
    print("-" * 120)
    
    # Sort by Macro F1 descending
    results_sorted = sorted(results, key=lambda x: x['Macro_F1'], reverse=True)
    
    for r in results_sorted:
        line = (f"{r['Config']:<25} {r['N_Samples']:>7} "
                f"{r['P_Precision']:>7.4f} {r['P_Recall']:>6.4f} {r['P_F1']:>6.4f} "
                f"{r['SB_Precision']:>8.4f} {r['SB_Recall']:>7.4f} {r['SB_F1']:>6.4f} "
                f"{r['Macro_F1']:>6.4f} {r['Accuracy']:>6.4f}")
        print(line)
    
    best = results_sorted[0]
    print(f"\nBEST CONFIG: {best['Config']} (Macro F1: {best['Macro_F1']:.4f})")

if __name__ == "__main__":
    main()
