import os
import sys
import json
import optuna
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_recall_curve, auc
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
from config import XGB_OUT_DIR, get_logger
from data_utils import get_stratified_group_split

logger = get_logger(__name__)

def objective(trial):
    # ---------- LOAD DATA ----------
    data_path = os.path.join(XGB_OUT_DIR, "xgboost_labeled_windows.csv")
    df = pd.read_csv(data_path).dropna(subset=['label'])
    
    TOP_N_FEATURES = 25
    banned_cols = ['lat', 'lon', 'suggestion_confidence', 'score']
    metadata_cols = ['event_id', 'trip_id', 'label', 'source', 'time_s', 'timestamp'] + banned_cols
    numeric_df = df.drop(columns=[c for c in metadata_cols if c in df.columns]).select_dtypes(include=[np.number])
    all_feature_cols = numeric_df.columns.tolist()

    df = df.dropna(subset=all_feature_cols)
    X_all = df[all_feature_cols].values
    y_all = df['label'].values

    # Pass 1: Quick Feature Selection
    le_fs = LabelEncoder()
    y_all_enc = le_fs.fit_transform(y_all)
    selector = XGBClassifier(n_estimators=50, max_depth=4, subsample=0.8,
                             colsample_bytree=0.8, random_state=42, n_jobs=1)
    selector.fit(X_all, y_all_enc)
    importances = selector.feature_importances_
    top_idx = np.argsort(importances)[::-1][:TOP_N_FEATURES]
    feature_cols = [all_feature_cols[i] for i in sorted(top_idx)]

    # Pass 2: Select Data
    X = df[feature_cols].values
    y = df['label'].values
    groups = df['trip_id'].values

    if 'source' in df.columns:
        source_values = df['source'].fillna('original').values
    else:
        source_values = np.array(['original'] * len(df))

    # Splitting
    dev_groups_list, _ = get_stratified_group_split(groups, y, train_ratio=0.8)
    dev_mask = np.isin(groups, dev_groups_list)
    
    X_dev = X[dev_mask]
    y_dev_raw = y[dev_mask]
    groups_dev = groups[dev_mask]
    source_dev = source_values[dev_mask]

    le = LabelEncoder()
    y_dev = le.fit_transform(y_dev_raw)
    classes = le.classes_
    
    if "Pothole" not in classes:
        # Failsafe if Pothole isn't in dev set
        return 0.0
    p_idx = list(classes).index("Pothole")

    # ---------- HYPERPARAMETER SEARCH SPACE ----------
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'max_depth': trial.suggest_int('max_depth', 3, 9),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 10.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 10.0),
        'random_state': 42,
        'n_jobs': 1
    }

    # ---------- CROSS-VALIDATION ----------
    sgkf = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=42)
    fold_prauc = []

    for fold, (train_idx, val_idx) in enumerate(sgkf.split(X_dev, y_dev, groups_dev)):
        X_train, y_train = X_dev[train_idx], y_dev[train_idx]
        
        # Filter augmented twins out of validation set
        is_original_val = np.array([not str(s).startswith('augmented') for s in source_dev[val_idx]])
        clean_val_idx = val_idx[is_original_val]
        X_val, y_val = X_dev[clean_val_idx], y_dev[clean_val_idx]

        missing_classes = set(range(len(classes))) - set(y_train)
        if missing_classes:
            for mc in missing_classes:
                X_train = np.vstack([X_train, X_train[0]])
                y_train = np.append(y_train, mc)

        model = XGBClassifier(**params)
        weights_train = compute_sample_weight('balanced', y_train)
        
        # Fit model
        model.fit(X_train, y_train, sample_weight=weights_train)
        
        # Predict on clean validation set
        y_proba = model.predict_proba(X_val)
        
        y_val_pothole_fold = (y_val == p_idx).astype(int)
        y_proba_pothole_fold = y_proba[:, p_idx]
        
        if sum(y_val_pothole_fold) == 0:
            continue
            
        prec, rec, _ = precision_recall_curve(y_val_pothole_fold, y_proba_pothole_fold)
        pr_auc_val = auc(rec, prec)
        fold_prauc.append(pr_auc_val)

    if len(fold_prauc) == 0:
        return 0.0
        
    avg_prauc = np.mean(fold_prauc)
    return avg_prauc

def main():
    logger.info("Starting XGBoost Hyperparameter Tuning with Optuna...")
    
    # Enable Optuna pruning and logging
    optuna.logging.set_verbosity(optuna.logging.INFO)
    
    # We want to maximize Average PR-AUC
    study = optuna.create_study(direction='maximize', study_name="XGBoost_PR_AUC_Tuning")
    study.optimize(objective, n_trials=50)

    logger.info("Number of finished trials: {}".format(len(study.trials)))
    logger.info("Best trial:")
    trial = study.best_trial

    logger.info("  Value (Average PR-AUC): {}".format(trial.value))
    logger.info("  Params: ")
    for key, value in trial.params.items():
        logger.info("    {}: {}".format(key, value))
        
    # Save best parameters to JSON
    _PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    model_dir = os.path.join(_PROJECT_ROOT, "evaluation", "models", "xgboost")
    os.makedirs(model_dir, exist_ok=True)
    
    best_params_path = os.path.join(model_dir, "best_params.json")
    with open(best_params_path, "w") as f:
        json.dump(trial.params, f, indent=4)
        
    logger.info(f"Best parameters saved to {best_params_path}")
    logger.info("You can now run train.py which will automatically use these best parameters!")

if __name__ == "__main__":
    main()
