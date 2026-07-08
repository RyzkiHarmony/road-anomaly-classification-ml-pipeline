# tune.py
# Skrip untuk pencarian hyperparameter XGBoost secara otomatis.

import os
import sys
import itertools
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import f1_score, precision_recall_curve
from xgboost import XGBClassifier

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
from config import XGB_OUT_DIR, get_logger

logger = get_logger(__name__)

def get_stratified_group_split(groups, y_raw, train_ratio=0.7):
    unique_classes = np.unique(y_raw)
    class_to_idx = {c: i for i, c in enumerate(unique_classes)}
    y_idx = np.array([class_to_idx[val] for val in y_raw])

    group_names = np.unique(groups)
    group_counts = {g: np.array([np.sum(y_idx[groups == g] == i) for i in range(len(unique_classes))]) for g in group_names}
    total_counts = np.sum(list(group_counts.values()), axis=0)

    train_groups = set()
    test_groups = set()
    current_train = np.zeros(len(unique_classes))

    minority_indices = [class_to_idx[c] for c in ['Pothole', 'Speed Bump'] if c in class_to_idx]
    sorted_groups = sorted(group_names, key=lambda g: np.sum(group_counts[g][minority_indices]), reverse=True)

    for g in sorted_groups:
        counts = group_counts[g]
        ratio_if_train = (current_train + counts) / (total_counts + 1e-9)
        err_train = np.sum((ratio_if_train - train_ratio) ** 2)
        
        ratio_if_test = current_train / (total_counts + 1e-9)
        err_test = np.sum((ratio_if_test - train_ratio) ** 2)
        
        if err_train < err_test:
            train_groups.add(g)
            current_train += counts
        else:
            test_groups.add(g)

    return list(train_groups), list(test_groups)

def main():
    data_path = os.path.join(XGB_OUT_DIR, "xgboost_labeled_windows.csv")
    if not os.path.exists(data_path):
        logger.error(f"Dataset not found at {data_path}. Please run build_xgboost_data.py first.")
        return

    logger.info(f"Loading dataset for tuning from {data_path}...")
    df = pd.read_csv(data_path)
    df = df.dropna(subset=['label'])
    
    TOP_N_FEATURES = 25
    banned_cols = ['lat', 'lon', 'suggestion_confidence', 'score']
    metadata_cols = ['event_id', 'trip_id', 'label', 'source', 'time_s', 'timestamp'] + banned_cols
    numeric_df = df.drop(columns=[c for c in metadata_cols if c in df.columns]).select_dtypes(include=[np.number])
    all_feature_cols = numeric_df.columns.tolist()
    
    df = df.dropna(subset=all_feature_cols)
    X_all = df[all_feature_cols].values
    y_all = df['label'].values
    
    logger.info(f"Feature Selection Pass 1: ranking {len(all_feature_cols)} features by XGBoost gain...")
    le_fs = LabelEncoder()
    y_all_enc = le_fs.fit_transform(y_all)
    selector = XGBClassifier(n_estimators=50, max_depth=4, subsample=0.8,
                             colsample_bytree=0.8, random_state=42, n_jobs=1)
    selector.fit(X_all, y_all_enc)
    importances = selector.feature_importances_
    top_idx = np.argsort(importances)[::-1][:TOP_N_FEATURES]
    feature_cols = [all_feature_cols[i] for i in sorted(top_idx)]
    
    logger.info(f"Top {TOP_N_FEATURES} features selected for tuning: {feature_cols}")
    
    X = df[feature_cols].values
    y_raw = df['label'].values
    groups = df['trip_id'].values
    source_values = df['source'].fillna('original').values if 'source' in df.columns else np.array(['original'] * len(df))

    # Split Dev Set
    dev_groups_list, _ = get_stratified_group_split(groups, y_raw, train_ratio=0.7)
    dev_mask = np.isin(groups, dev_groups_list)

    X_dev = X[dev_mask]
    y_dev_raw = y_raw[dev_mask]
    groups_dev = groups[dev_mask]
    source_dev = source_values[dev_mask]

    le = LabelEncoder()
    y_dev = le.fit_transform(y_dev_raw)
    classes = le.classes_
    p_idx = list(classes).index("Pothole")
    sb_idx = list(classes).index("Speed Bump") if "Speed Bump" in list(classes) else -1
    non_event_idx = list(classes).index("Non-Event") if "Non-Event" in classes else 0

    # Hyperparameter search space definition
    param_grid = {
        'max_depth': [2, 3, 4],
        'learning_rate': [0.01, 0.05, 0.1],
        'n_estimators': [40, 60, 100],
        'min_child_weight': [1, 5, 10],
        'reg_lambda': [1.0, 5.0, 10.0]
    }

    # Custom Grid Search with StratifiedGroupKFold (3-fold)
    sgkf = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42)
    
    best_score = -1.0
    best_params = None
    
    keys, values = zip(*param_grid.items())
    experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    logger.info(f"Running sweep over {len(experiments)} hyperparameter combinations...")
    
    for idx, params in enumerate(experiments):
        oof_y_true = []
        oof_y_proba = []
        
        for fold, (train_idx, val_idx) in enumerate(sgkf.split(X_dev, y_dev, groups_dev)):
            X_train, y_train = X_dev[train_idx], y_dev[train_idx]
            
            # Validation Set clean (original only)
            is_original_val = np.array([not str(s).startswith('augmented') for s in source_dev[val_idx]])
            clean_val_idx = val_idx[is_original_val]
            X_val, y_val = X_dev[clean_val_idx], y_dev[clean_val_idx]

            # Inject missing classes in training if any
            missing_classes = set(range(len(classes))) - set(y_train)
            if missing_classes:
                for mc in missing_classes:
                    global_idx = np.where(y_dev == mc)[0][0]
                    X_train = np.vstack([X_train, X_dev[global_idx]])
                    y_train = np.append(y_train, mc)

            model = XGBClassifier(
                n_estimators=params['n_estimators'],
                max_depth=params['max_depth'],
                learning_rate=params['learning_rate'],
                min_child_weight=params['min_child_weight'],
                reg_lambda=params['reg_lambda'],
                subsample=0.7,
                colsample_bytree=0.7,
                random_state=42,
                n_jobs=1
            )
            
            weights_train = compute_sample_weight('balanced', y_train)
            model.fit(X_train, y_train, sample_weight=weights_train)
            
            y_proba = model.predict_proba(X_val)
            oof_y_true.extend(y_val)
            oof_y_proba.extend(y_proba)
            
        oof_y_true = np.array(oof_y_true)
        oof_y_proba = np.array(oof_y_proba)
        
        # Evaluate using Default Argmax
        oof_y_pred = np.argmax(oof_y_proba, axis=1)
        best_f1_pothole = f1_score(oof_y_true, oof_y_pred, labels=[p_idx], average='macro', zero_division=0)
        
        best_f1_sb = 0.0
        if sb_idx != -1:
            best_f1_sb = f1_score(oof_y_true, oof_y_pred, labels=[sb_idx], average='macro', zero_division=0)

        # Objective function: Prioritize Pothole F1 score while maintaining some Speed Bump F1 score
        composite_score = 0.7 * best_f1_pothole + 0.3 * best_f1_sb
        
        if (idx + 1) % 20 == 0 or idx == 0:
            logger.info(f"Exp {idx+1}/{len(experiments)} | Params: {params} | Pothole F1: {best_f1_pothole:.4f} | SB F1: {best_f1_sb:.4f} | Composite: {composite_score:.4f}")
            
        if composite_score > best_score:
            best_score = composite_score
            best_f1_p = best_f1_pothole
            best_f1_s = best_f1_sb
            best_params = params
            
    print("\n" + "=" * 60)
    print("                 BEST XGBOOST HYPERPARAMETERS                ")
    print("=" * 60)
    print(f"Best Params: {best_params}")
    print(f"Best Composite Score: {best_score:.4f}")
    print(f"Best OOF Pothole F1 (Default Argmax): {best_f1_p:.4f}")
    print(f"Best OOF Speed Bump F1 (Default Argmax): {best_f1_s:.4f}")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    main()
