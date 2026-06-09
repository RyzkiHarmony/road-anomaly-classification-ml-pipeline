import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import StratifiedGroupKFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_recall_curve, make_scorer
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

from config import OUT_FOLDER, get_logger, BEST_FEATURES

logger = get_logger(__name__)

def evaluate_model_cv(model, X, y_enc, groups, source_values, p_idx, cv_strategy, model_name, classes):
    print(f"\n--- Evaluating {model_name} ---")
    fold_f1 = []
    oof_y_true = []
    oof_y_pred = []
    
    for fold, (train_idx, test_idx) in enumerate(cv_strategy.split(X, y_enc, groups)):
        X_train, y_train = X[train_idx], y_enc[train_idx]
        
        is_original_test = np.array([not str(s).startswith('augmented') for s in source_values[test_idx]])
        clean_test_idx = test_idx[is_original_test]
        X_test, y_test = X[clean_test_idx], y_enc[clean_test_idx]

        missing_classes = set(range(3)) - set(y_train)
        if missing_classes:
            for mc in missing_classes:
                global_idx = np.where(y_enc == mc)[0][0]
                X_train = np.vstack([X_train, X[global_idx]])
                y_train = np.append(y_train, mc)
                
        try:
            model.fit(X_train, y_train)
        except Exception as e:
            logger.error(f"Fit failed for {model_name} on fold {fold+1}: {e}")
            fold_f1.append(0)
            continue
            
        y_pred = model.predict(X_test)
        
        oof_y_true.extend(y_test)
        oof_y_pred.extend(y_pred)
        
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        p_f1 = report.get(str(p_idx), {}).get('f1-score', 0)
        fold_f1.append(p_f1)
        print(f"Fold {fold+1} F1: {p_f1:.4f}")
        
    avg_f1 = np.mean(fold_f1)
    print(f"\n{model_name} - OOF Classification Report:")
    print(classification_report(oof_y_true, oof_y_pred, target_names=classes, zero_division=0))
    print(f"{model_name} - Avg Pothole F1: {avg_f1:.4f}")
    return avg_f1, oof_y_true, oof_y_pred

def main():
    data_path = os.path.join(OUT_FOLDER, "manual_labeled_windows.csv")
    if not os.path.exists(data_path): return
    df = pd.read_csv(data_path).dropna(subset=['label'])
    
    feature_cols = [c for c in df.columns if c in BEST_FEATURES]
    df = df.dropna(subset=feature_cols)
    
    X = df[feature_cols].values
    y = df['label'].values
    groups = df['trip_id'].values
    
    source_values = df['source'].fillna('original').values if 'source' in df.columns else np.array(['original'] * len(df))

    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    classes = le.classes_
    p_idx = list(classes).index("Pothole")

    cv_strategy = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    
    print("\n" + "="*60)
    print("PHASE 1: BASELINE EVALUATION (Default Parameters)")
    print("="*60)
    
    from sklearn.pipeline import Pipeline
    
    # SENIOR ML ENGINEER FIX: SMOTE dihapus. Menggabungkan dua fitur pothole 
    # menciptakan data fisika hantu yang tidak pernah ada di dunia nyata.
    # Nama variabel tetap 'smote_rf' agar tidak merusak baris kode di bawahnya.
    smote_rf = Pipeline([
        ('clf', RandomForestClassifier(random_state=42, n_jobs=-1, class_weight='balanced_subsample'))
    ])
    
    smote_xgb = Pipeline([
        ('clf', XGBClassifier(random_state=42, eval_metric='mlogloss', n_jobs=-1))
    ])
    
    rf_base_f1, _, _ = evaluate_model_cv(smote_rf, X, y_enc, groups, source_values, p_idx, cv_strategy, "Random Forest (Baseline)", classes)
    xgb_base_f1, _, _ = evaluate_model_cv(smote_xgb, X, y_enc, groups, source_values, p_idx, cv_strategy, "XGBoost (Baseline)", classes)
    
    print("\n" + "="*60)
    print("PHASE 2: HYPERPARAMETER TUNING (RandomizedSearchCV)")
    print("="*60)
    
    def pothole_f1_scorer(y_true, y_pred):
        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        return report.get(str(p_idx), {}).get('f1-score', 0)
        
    scorer = make_scorer(pothole_f1_scorer)
    
    custom_cv = []
    for train_idx, test_idx in cv_strategy.split(X, y_enc, groups):
        is_original_test = np.array([not str(s).startswith('augmented') for s in source_values[test_idx]])
        clean_test_idx = test_idx[is_original_test]
        custom_cv.append((train_idx, clean_test_idx))

    rf_param_dist = {
        'clf__n_estimators': [100, 200, 300],
        'clf__max_depth': [5, 10, None],
        'clf__min_samples_leaf': [1, 3, 5],
        'clf__class_weight': ['balanced', 'balanced_subsample']
    }
    logger.info("Tuning Random Forest...")
    rf_random = RandomizedSearchCV(estimator=smote_rf, param_distributions=rf_param_dist, 
                                   n_iter=10, cv=custom_cv, scoring=scorer, random_state=42, n_jobs=-1, verbose=1)
    rf_random.fit(X, y_enc)
    best_rf = rf_random.best_estimator_
    print(f"Best RF Params: {rf_random.best_params_}")
    
    xgb_param_dist = {
        'clf__n_estimators': [100, 200, 300],
        'clf__max_depth': [3, 5, 7],
        'clf__learning_rate': [0.01, 0.05, 0.1],
        'clf__subsample': [0.8, 1.0],
        'clf__colsample_bytree': [0.8, 1.0]
    }
    logger.info("Tuning XGBoost...")
    xgb_random = RandomizedSearchCV(estimator=smote_xgb, param_distributions=xgb_param_dist, 
                                    n_iter=10, cv=custom_cv, scoring=scorer, random_state=42, n_jobs=-1, verbose=1)
    xgb_random.fit(X, y_enc)
    best_xgb = xgb_random.best_estimator_
    print(f"Best XGB Params: {xgb_random.best_params_}")
    
    print("\n" + "="*60)
    print("PHASE 3: TUNED EVALUATION")
    print("="*60)
    
    rf_tuned_f1, _, _ = evaluate_model_cv(best_rf, X, y_enc, groups, source_values, p_idx, cv_strategy, "Random Forest (Tuned)", classes)
    xgb_tuned_f1, _, _ = evaluate_model_cv(best_xgb, X, y_enc, groups, source_values, p_idx, cv_strategy, "XGBoost (Tuned)", classes)
    
    print("\n" + "="*60)
    print("SUMMARY OF RESULTS (OOF Pothole F1)")
    print("="*60)
    print(f"RF Baseline:   {rf_base_f1:.4f}")
    print(f"RF Tuned:      {rf_tuned_f1:.4f}")
    print(f"XGB Baseline:  {xgb_base_f1:.4f}")
    print(f"XGB Tuned:     {xgb_tuned_f1:.4f}")
    
    winner_name = "Random Forest"
    winner_model = best_rf
    winner_f1 = rf_tuned_f1
    if xgb_tuned_f1 > rf_tuned_f1:
        winner_name = "XGBoost"
        winner_model = best_xgb
        winner_f1 = xgb_tuned_f1
        
    print(f"\nWINNER: {winner_name} with F1-Score: {winner_f1:.4f}")
    


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    main()
