import sys
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from config import OUT_FOLDER, BEST_FEATURES

def run_cv(df, feature_cols, y_enc, groups, source_values, classes, use_smote=True):
    """Run 5-fold Stratified Group K-Fold CV with or without SMOTE."""
    X = df[feature_cols].values
    
    unique_groups = len(np.unique(groups))
    n_folds = min(5, unique_groups) if unique_groups >= 2 else 5
    cv_strategy = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    oof_y_true = []
    oof_y_pred = []
    fold_metrics = []
    
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
            except Exception as e:
                pass

        model = RandomForestClassifier(
            n_estimators=300, max_depth=5, min_samples_leaf=3,
            class_weight='balanced_subsample', random_state=42, n_jobs=-1
        )
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        oof_y_true.extend(y_test)
        oof_y_pred.extend(y_pred)
        
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        fold_metrics.append(report)
    
    oof_y_true = np.array(oof_y_true)
    oof_y_pred = np.array(oof_y_pred)
    
    return oof_y_true, oof_y_pred

def main():
    data_path = os.path.join(OUT_FOLDER, "manual_labeled_windows.csv")
    df = pd.read_csv(data_path).dropna(subset=['label'])
    
    feature_cols = [c for c in df.columns if c in BEST_FEATURES]
    df = df.dropna(subset=feature_cols)

    y = df['label'].values
    groups = df['trip_id'].values
    
    if 'source' in df.columns:
        source_values = df['source'].fillna('original').values
    else:
        source_values = np.array(['original'] * len(df))

    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    classes = list(le.classes_)

    # ========== Experiment A: WITH SMOTE ==========
    print("=" * 60)
    print("EXPERIMENT A: WITH SMOTE (Current Pipeline)")
    print("=" * 60)
    oof_true_smote, oof_pred_smote = run_cv(df, feature_cols, y_enc, groups, source_values, classes, use_smote=True)
    print(classification_report(oof_true_smote, oof_pred_smote, target_names=classes, zero_division=0))

    # ========== Experiment B: WITHOUT SMOTE ==========
    print("=" * 60)
    print("EXPERIMENT B: WITHOUT SMOTE (class_weight only)")
    print("=" * 60)
    oof_true_no, oof_pred_no = run_cv(df, feature_cols, y_enc, groups, source_values, classes, use_smote=False)
    print(classification_report(oof_true_no, oof_pred_no, target_names=classes, zero_division=0))

    # ========== Head-to-Head Summary ==========
    r_smote = classification_report(oof_true_smote, oof_pred_smote, target_names=classes, output_dict=True, zero_division=0)
    r_no = classification_report(oof_true_no, oof_pred_no, target_names=classes, output_dict=True, zero_division=0)
    
    print("=" * 60)
    print("HEAD-TO-HEAD COMPARISON")
    print("=" * 60)
    print(f"{'Metric':<25} {'WITH SMOTE':>12} {'WITHOUT SMOTE':>15} {'Delta':>10}")
    print("-" * 62)
    
    for cls_name in classes:
        for metric in ['precision', 'recall', 'f1-score']:
            v_s = r_smote[cls_name][metric]
            v_n = r_no[cls_name][metric]
            delta = v_s - v_n
            label = f"{cls_name} {metric}"
            print(f"{label:<25} {v_s:>12.4f} {v_n:>15.4f} {delta:>+10.4f}")
        print()
    
    for avg in ['macro avg', 'weighted avg']:
        for metric in ['precision', 'recall', 'f1-score']:
            v_s = r_smote[avg][metric]
            v_n = r_no[avg][metric]
            delta = v_s - v_n
            label = f"{avg} {metric}"
            print(f"{label:<25} {v_s:>12.4f} {v_n:>15.4f} {delta:>+10.4f}")
    
    print(f"\n{'accuracy':<25} {r_smote['accuracy']:>12.4f} {r_no['accuracy']:>15.4f} {r_smote['accuracy']-r_no['accuracy']:>+10.4f}")

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore')
    main()
