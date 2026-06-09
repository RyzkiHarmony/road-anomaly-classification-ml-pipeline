import sys
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report

# Add parent directory to sys.path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from config import OUT_FOLDER, BEST_FEATURES

# Candidate pool: BEST_FEATURES + 3 new shape-aware features
CANDIDATE_FEATURES = BEST_FEATURES + [
    "rise_time_ratio",
    "peak_asymmetry",
    "waveform_complexity",
]

def evaluate_feature_set(feature_list, df, y_enc, groups, source_values, classes, p_idx, sb_idx):
    """Run 5-fold Stratified Group K-Fold CV and return avg F1 for Pothole and Speed Bump."""
    X_curr = df[feature_list].values
    
    unique_groups = len(np.unique(groups))
    n_folds = min(5, unique_groups) if unique_groups >= 2 else 5
    cv_strategy = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    fold_p_f1 = []
    fold_sb_f1 = []
    fold_macro_f1 = []
    
    for fold, (train_idx, test_idx) in enumerate(cv_strategy.split(X_curr, y_enc, groups)):
        X_train, y_train = X_curr[train_idx], y_enc[train_idx]
        
        is_original_test = np.array([not str(s).startswith('augmented') for s in source_values[test_idx]])
        clean_test_idx = test_idx[is_original_test]
        X_test, y_test = X_curr[clean_test_idx], y_enc[clean_test_idx]

        missing_classes = set(range(len(classes))) - set(y_train)
        if missing_classes:
            for mc in missing_classes:
                global_idx = np.where(y_enc == mc)[0][0]
                X_train = np.vstack([X_train, X_curr[global_idx]])
                y_train = np.append(y_train, mc)

        try:
            smote = SMOTE(random_state=42, k_neighbors=min(5, min(np.bincount(y_train)) - 1))
            if min(np.bincount(y_train)) > 1:
                X_train, y_train = smote.fit_resample(X_train, y_train)
        except:
            pass

        model = RandomForestClassifier(n_estimators=300, max_depth=5, min_samples_leaf=3, class_weight='balanced_subsample', random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        
        fold_p_f1.append(report.get(str(p_idx), {}).get('f1-score', 0))
        fold_sb_f1.append(report.get(str(sb_idx), {}).get('f1-score', 0))
        fold_macro_f1.append(report.get('macro avg', {}).get('f1-score', 0))
        
    return np.mean(fold_p_f1), np.mean(fold_sb_f1), np.mean(fold_macro_f1)

def main():
    data_path = os.path.join(OUT_FOLDER, "manual_labeled_windows.csv")
    df = pd.read_csv(data_path).dropna(subset=['label'])
    
    # Use ALL candidate features that exist in the CSV
    available_features = [c for c in CANDIDATE_FEATURES if c in df.columns]
    df = df.dropna(subset=available_features)

    X = df[available_features].values
    y = df['label'].values
    groups = df['trip_id'].values
    
    if 'source' in df.columns:
        source_values = df['source'].fillna('original').values
    else:
        source_values = np.array(['original'] * len(df))

    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    classes = list(le.classes_)
    p_idx = classes.index("Pothole")
    sb_idx = classes.index("Speed Bump")

    # ========== STEP 1: Feature Importance Ranking ==========
    print("\n[STEP 1] Menghitung Feature Importance (semua kandidat)...")
    smote_global = SMOTE(random_state=42, k_neighbors=min(5, min(np.bincount(y_enc)) - 1))
    X_sm, y_sm = smote_global.fit_resample(X, y_enc)
    
    base_rf = RandomForestClassifier(n_estimators=300, max_depth=5, min_samples_leaf=3, class_weight='balanced_subsample', random_state=42, n_jobs=-1)
    base_rf.fit(X_sm, y_sm)
    
    importances = base_rf.feature_importances_
    feat_imp = sorted(zip(available_features, importances), key=lambda x: x[1], reverse=True)
    
    print(f"\n--- Feature Importance Ranking ({len(available_features)} fitur) ---")
    for rank, (f, imp) in enumerate(feat_imp, 1):
        tag = " [SHAPE-AWARE]" if f in ("rise_time_ratio", "peak_asymmetry", "waveform_complexity") else ""
        print(f"#{rank:2} {f:25}: {imp:.4f}{tag}")
        
    ordered_features = [f[0] for f in feat_imp]

    # ========== STEP 2: Ablation Study ==========
    print("\n[STEP 2] Feature Ablation Study (Cross-Validation)")
    
    results = []
    total = len(ordered_features)
    test_counts = sorted(set([total, 15, 12, 10, 7, 5, 3]), reverse=True)
    test_counts = [c for c in test_counts if c <= total]
    
    for count in test_counts:
        current_features = ordered_features[:count]
        avg_p, avg_sb, avg_macro = evaluate_feature_set(current_features, df, y_enc, groups, source_values, classes, p_idx, sb_idx)
        
        results.append({
            'N_Features': count,
            'Pothole_F1': avg_p,
            'SpeedBump_F1': avg_sb,
            'Macro_F1': avg_macro
        })
        print(f"Top {count:2} Fitur -> Pothole F1: {avg_p:.4f} | Speed Bump F1: {avg_sb:.4f} | Macro F1: {avg_macro:.4f}")

    # ========== STEP 3: Head-to-Head (Old 15 vs New 18) ==========
    print("\n[STEP 3] Head-to-Head: Old BEST_FEATURES (15) vs All Candidates (18)")
    
    old_features = [c for c in BEST_FEATURES if c in df.columns]
    avg_p_old, avg_sb_old, avg_macro_old = evaluate_feature_set(old_features, df, y_enc, groups, source_values, classes, p_idx, sb_idx)
    print(f"OLD (15 fitur): Pothole F1: {avg_p_old:.4f} | Speed Bump F1: {avg_sb_old:.4f} | Macro F1: {avg_macro_old:.4f}")
    
    avg_p_new, avg_sb_new, avg_macro_new = evaluate_feature_set(available_features, df, y_enc, groups, source_values, classes, p_idx, sb_idx)
    print(f"NEW (18 fitur): Pothole F1: {avg_p_new:.4f} | Speed Bump F1: {avg_sb_new:.4f} | Macro F1: {avg_macro_new:.4f}")

    delta_p = avg_p_new - avg_p_old
    delta_sb = avg_sb_new - avg_sb_old
    print(f"\nDelta Pothole: {delta_p:+.4f} | Delta Speed Bump: {delta_sb:+.4f}")

    # ========== Plot ==========
    res_df = pd.DataFrame(results)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(res_df['N_Features'], res_df['Pothole_F1'], marker='o', linewidth=2, label='Pothole F1', color='#e74c3c')
    ax.plot(res_df['N_Features'], res_df['SpeedBump_F1'], marker='s', linewidth=2, label='Speed Bump F1', color='#3498db')
    ax.plot(res_df['N_Features'], res_df['Macro_F1'], marker='^', linewidth=2, label='Macro F1', color='#2ecc71', linestyle='--')
    ax.set_title('Feature Ablation Study (with Shape-Aware Features)')
    ax.set_xlabel('Jumlah Fitur (ranked by importance)')
    ax.set_ylabel('Out-of-Fold F1-Score')
    ax.set_xticks(test_counts)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend()
    
    save_path = os.path.join(OUT_FOLDER, "ablation_study_v2.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n[SELESAI] Grafik disimpan di {save_path}")

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore')
    main()
