import os
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedGroupKFold
import json
from sklearn.utils.class_weight import compute_sample_weight

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(BASE_DIR, "src", "utils"))

from config import XGB_OUT_DIR, CNN_OUT_DIR, get_logger

logger = get_logger(__name__)

XGB_MODEL_DIR = os.path.join(BASE_DIR, "evaluation", "models", "xgboost")
CNN_MODEL_DIR = os.path.join(BASE_DIR, "evaluation", "models", "cnn_1d")

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

def analyze_xgb_fn():
    logger.info("=" * 60)
    logger.info("  XGBoost Error Audit: False Negative Analysis")
    logger.info("=" * 60)

    data_path = os.path.join(XGB_OUT_DIR, "xgboost_labeled_windows.csv")
    df = pd.read_csv(data_path).dropna(subset=['label'])

    with open(os.path.join(XGB_MODEL_DIR, "xgboost_features.json"), "r") as f:
        feature_cols = json.load(f)

    df = df.dropna(subset=feature_cols)

    X = df[feature_cols].values
    y_raw = df['label'].values
    groups = df['trip_id'].values
    source_values = df['source'].fillna('original').values if 'source' in df.columns else np.array(['original'] * len(df))

    dev_groups_list, test_groups_list = get_stratified_group_split(groups, y_raw, train_ratio=0.7)
    dev_mask = np.isin(groups, dev_groups_list)

    X_dev = X[dev_mask]
    y_dev_raw = y_raw[dev_mask]
    groups_dev = groups[dev_mask]
    source_dev = source_values[dev_mask]
    df_dev = df[dev_mask].reset_index(drop=True)

    le = LabelEncoder()
    y_dev = le.fit_transform(y_dev_raw)
    classes = le.classes_
    p_idx = list(classes).index("Pothole")
    sb_idx = list(classes).index("Speed Bump") if "Speed Bump" in list(classes) else -1
    non_event_idx = list(classes).index("Non-Event") if "Non-Event" in classes else 0

    from xgboost import XGBClassifier
    sgkf = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42)
    oof_indices = []
    oof_y_true = []
    oof_y_pred = []
    oof_y_proba = []

    for fold, (train_idx, val_idx) in enumerate(sgkf.split(X_dev, y_dev, groups_dev)):
        X_train, y_train = X_dev[train_idx], y_dev[train_idx]
        is_original_val = np.array([not str(s).startswith('augmented') for s in source_dev[val_idx]])
        clean_val_idx = val_idx[is_original_val]
        X_val, y_val = X_dev[clean_val_idx], y_dev[clean_val_idx]

        missing_classes = set(range(len(classes))) - set(y_train)
        if missing_classes:
            for mc in missing_classes:
                global_idx = np.where(y_dev == mc)[0][0]
                X_train = np.vstack([X_train, X_dev[global_idx]])
                y_train = np.append(y_train, mc)

        model = XGBClassifier(
            n_estimators=100, max_depth=4, min_child_weight=1,
            learning_rate=0.1, subsample=0.7, colsample_bytree=0.7,
            reg_lambda=1.0, reg_alpha=1.0, random_state=42, n_jobs=1
        )
        weights_train = compute_sample_weight('balanced', y_train)
        model.fit(X_train, y_train, sample_weight=weights_train)

        y_proba = model.predict_proba(X_val)
        y_pred = np.argmax(y_proba, axis=1)

        oof_indices.extend(clean_val_idx)
        oof_y_true.extend(y_val)
        oof_y_pred.extend(y_pred)
        oof_y_proba.extend(y_proba)

    oof_indices = np.array(oof_indices)
    oof_y_true = np.array(oof_y_true)
    oof_y_pred = np.array(oof_y_pred)
    oof_y_proba = np.array(oof_y_proba)

    fn_p_mask = (oof_y_pred == non_event_idx) & (oof_y_true == p_idx)
    fn_sb_mask = (oof_y_pred == non_event_idx) & (oof_y_true == sb_idx)

    print(f"\nXGBoost False Negatives (Actual Event -> Predicted Non-Event):")
    print(f"FN Pothole (Actual Pothole -> Pred Non-Event): {fn_p_mask.sum()}")
    print(f"FN Speed Bump (Actual Speed Bump -> Pred Non-Event): {fn_sb_mask.sum()}")

    if fn_p_mask.sum() > 0:
        probs = oof_y_proba[fn_p_mask]
        print(f"\nFN Pothole Probability Analysis:")
        print(f"Mean Prob Non-Event: {probs[:, non_event_idx].mean():.4f}")
        print(f"Mean Prob Pothole: {probs[:, p_idx].mean():.4f}")
        
    if fn_sb_mask.sum() > 0:
        probs = oof_y_proba[fn_sb_mask]
        print(f"\nFN Speed Bump Probability Analysis:")
        print(f"Mean Prob Non-Event: {probs[:, non_event_idx].mean():.4f}")
        print(f"Mean Prob Speed Bump: {probs[:, sb_idx].mean():.4f}")

def analyze_cnn_fn():
    logger.info("\n" + "=" * 60)
    logger.info("  1D-CNN Error Audit: False Negative Analysis")
    logger.info("=" * 60)

    try:
        oof_y_true = np.load(os.path.join(CNN_MODEL_DIR, "cnn_1d_oof_y_true.npy"))
        oof_y_pred = np.load(os.path.join(CNN_MODEL_DIR, "cnn_1d_oof_y_pred.npy"))
        oof_y_proba = np.load(os.path.join(CNN_MODEL_DIR, "cnn_1d_oof_y_proba.npy"))
        classes = np.load(os.path.join(CNN_MODEL_DIR, "cnn_1d_classes.npy"), allow_pickle=True)
    except FileNotFoundError:
        print("CNN OOF predictions not found.")
        return

    p_idx = list(classes).index("Pothole")
    sb_idx = list(classes).index("Speed Bump") if "Speed Bump" in list(classes) else -1
    non_event_idx = list(classes).index("Non-Event") if "Non-Event" in classes else 0

    fn_p_mask = (oof_y_pred == non_event_idx) & (oof_y_true == p_idx)
    fn_sb_mask = (oof_y_pred == non_event_idx) & (oof_y_true == sb_idx)

    print(f"\nCNN False Negatives (Actual Event -> Predicted Non-Event):")
    print(f"FN Pothole (Actual Pothole -> Pred Non-Event): {fn_p_mask.sum()}")
    print(f"FN Speed Bump (Actual Speed Bump -> Pred Non-Event): {fn_sb_mask.sum()}")

    if fn_p_mask.sum() > 0:
        probs = oof_y_proba[fn_p_mask]
        print(f"\nFN Pothole Probability Analysis:")
        print(f"Mean Prob Non-Event: {probs[:, non_event_idx].mean():.4f}")
        print(f"Mean Prob Pothole: {probs[:, p_idx].mean():.4f}")
        
    if fn_sb_mask.sum() > 0:
        probs = oof_y_proba[fn_sb_mask]
        print(f"\nFN Speed Bump Probability Analysis:")
        print(f"Mean Prob Non-Event: {probs[:, non_event_idx].mean():.4f}")
        print(f"Mean Prob Speed Bump: {probs[:, sb_idx].mean():.4f}")

if __name__ == "__main__":
    analyze_xgb_fn()
    analyze_cnn_fn()
