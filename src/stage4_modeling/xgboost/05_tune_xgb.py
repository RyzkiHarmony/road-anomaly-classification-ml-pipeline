import json
import os
import sys

import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import auc, precision_recall_curve
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.join(_PROJECT_ROOT, "src", "utils"))
sys.path.append(os.path.join(_PROJECT_ROOT, "src", "stage3_eda_and_splitting"))
sys.path.append(os.path.dirname(__file__))

from config import XGB_OUT_DIR, get_logger
from data_splitting import get_stratified_group_split

logger = get_logger(__name__)


def _select_non_redundant_features(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    top_n: int,
    num_classes: int,
    corr_threshold: float = 0.75
) -> tuple[list[str], list[dict]]:
    """
    Rank features using Dev split and eliminate multicollinear (redundant) features where |r| > corr_threshold.
    Returns (selected_features, dropped_redundant_info).
    """
    selector = XGBClassifier(
        n_estimators=50,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='multi:softprob',
        num_class=int(num_classes),
        random_state=42,
        n_jobs=1,
    )
    selector.fit(X, y)
    importances = selector.feature_importances_
    ranked_indices = np.argsort(importances)[::-1]
    ranked_features = [feature_names[i] for i in ranked_indices]

    df_features = pd.DataFrame(X, columns=feature_names)
    corr_matrix = df_features.corr().abs()

    selected_features = []
    dropped_info = []

    for feat in ranked_features:
        is_redundant = False
        for kept_feat in selected_features:
            r_val = corr_matrix.loc[feat, kept_feat]
            if r_val > corr_threshold:
                is_redundant = True
                dropped_info.append({
                    "dropped_feature": feat,
                    "redundant_with": kept_feat,
                    "correlation": float(r_val)
                })
                break
        if not is_redundant:
            selected_features.append(feat)

    final_features = [f for f in feature_names if f in selected_features[:top_n]]
    return final_features, dropped_info


def create_objective(X_dev, y_dev, groups_dev, source_dev, classes, p_idx):
    def objective(trial):
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
            'n_jobs': 1,
            'objective': 'multi:softprob',
            'num_class': int(len(classes)),
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

    return objective


def main():
    # ---------- LOAD DATA ----------
    data_path = os.path.join(XGB_OUT_DIR, "xgboost_labeled_windows.csv")
    if not os.path.exists(data_path):
        print(f"Error: Dataset tidak ditemukan di {data_path}. Silakan jalankan ekstraksi fitur terlebih dahulu.", flush=True)
        return

    print("\n" + "=" * 78, flush=True)
    print("        OPTIMASI HYPERPARAMETER XGBOOST (OPTUNA BAYESIAN OPTIMIZATION)", flush=True)
    print("=" * 78, flush=True)

    df = pd.read_csv(data_path).dropna(subset=['label'])

    TOP_N_FEATURES = 25
    banned_cols = ['lat', 'lon', 'suggestion_confidence', 'score']
    metadata_cols = ['event_id', 'trip_id', 'label', 'source', 'time_s', 'timestamp'] + banned_cols
    numeric_df = df.drop(columns=[c for c in metadata_cols if c in df.columns]).select_dtypes(include=[np.number])
    all_feature_cols = numeric_df.columns.tolist()

    df = df.dropna(subset=all_feature_cols)
    X_all = df[all_feature_cols].values
    y_all = df['label'].values
    groups = df['trip_id'].values

    if 'source' in df.columns:
        source_values = df['source'].fillna('original').values
    else:
        source_values = np.array(['original'] * len(df))

    # Split Dev Set (80%) and Holdout Test Set (20%) strictly by trip
    dev_groups_list, _ = get_stratified_group_split(groups, y_all, train_ratio=0.8)
    dev_mask = np.isin(groups, dev_groups_list)

    X_dev_full = X_all[dev_mask]
    y_dev_raw = y_all[dev_mask]
    groups_dev = groups[dev_mask]
    source_dev = source_values[dev_mask]

    le = LabelEncoder()
    y_dev = le.fit_transform(y_dev_raw)
    classes = le.classes_

    if "Pothole" not in classes:
        print("Error: Kelas 'Pothole' tidak ditemukan dalam Dev set.", flush=True)
        return
    p_idx = list(classes).index("Pothole")

    N_TRIALS = 100
    CORR_THRESHOLD = 0.75

    print("\n[1/4] PERSIAPAN DATASET & RUANG PENCARIAN (Dev Set 80%):", flush=True)
    print(f"  - Data Latih Tuning        : {len(dev_groups_list)} rute perjalanan | {len(X_dev_full):,} sampel", flush=True)
    print(f"  - Evaluasi Validasi        : 4-Fold Stratified Group K-Fold (Trip-Isolated)", flush=True)
    print(f"  - Target Optimasi          : Maksimalisasi Rata-rata PR-AUC (Kelas Pothole)", flush=True)
    print(f"  - Total Percobaan (Trials) : {N_TRIALS} iterasi", flush=True)

    # Feature selection with multicollinearity pruning on Dev Set only
    feature_cols, dropped_redundant = _select_non_redundant_features(
        X_dev_full, y_dev, all_feature_cols, TOP_N_FEATURES, len(classes), corr_threshold=CORR_THRESHOLD
    )

    print(f"\n[2/4] SELEKSI FITUR NON-REDUNDAN (Dev Set Only):", flush=True)
    print(f"  - Ambang Multikolinieritas : |r| <= {CORR_THRESHOLD}", flush=True)
    print(f"  - Total Fitur Kandidat     : {len(all_feature_cols)} fitur", flush=True)
    print(f"  - Fitur Dieliminasi        : {len(dropped_redundant)} fitur redundan (|r| > {CORR_THRESHOLD})", flush=True)
    print(f"  - Fitur Terpilih (Final)   : {len(feature_cols)} fitur independen", flush=True)
    print(f"  - Top 5 Fitur Terkuat      : {', '.join(feature_cols[:5])}...", flush=True)

    X_dev = pd.DataFrame(X_dev_full, columns=all_feature_cols)[feature_cols].values

    # Silence Optuna internal verbose logs
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    print(f"\n[3/4] PROSES OPTIMASI ({N_TRIALS} TRIALS):", flush=True)
    print("  " + "-" * 74, flush=True)
    print(f"  {'Trial':<8} {'Status':<16} {'PR-AUC (Pothole)':<20} {'Best PR-AUC':<14} {'Catatan':<16}", flush=True)
    print("  " + "-" * 74, flush=True)

    best_value = -1.0

    def print_callback(study, trial):
        nonlocal best_value
        val = trial.value if trial.value is not None else 0.0
        trial_num = trial.number + 1
        is_best = val > best_value
        is_checkpoint = (trial_num % 10 == 0) or (trial_num == 1) or (trial_num == N_TRIALS)

        if is_best:
            best_value = val
            print(f"  #{trial_num:<7} {'Selesai':<16} {val:<20.4f} {best_value:<14.4f} * Rekor Baru", flush=True)
        elif is_checkpoint:
            print(f"  #{trial_num:<7} {'Kemajuan':<16} {val:<20.4f} {best_value:<14.4f} Iterasi ke-{trial_num}", flush=True)

    objective = create_objective(X_dev, y_dev, groups_dev, source_dev, classes, p_idx)
    study = optuna.create_study(direction='maximize', study_name="XGBoost_PR_AUC_Tuning")
    study.optimize(objective, n_trials=N_TRIALS, callbacks=[print_callback])
    print("  " + "-" * 74, flush=True)

    trial = study.best_trial

    search_spaces = {
        'n_estimators': '[50, 300] (Integer)',
        'max_depth': '[3, 9] (Integer)',
        'min_child_weight': '[1, 10] (Integer)',
        'learning_rate': '[0.01, 0.30] (Log Float)',
        'subsample': '[0.50, 1.00] (Float)',
        'colsample_bytree': '[0.50, 1.00] (Float)',
        'reg_alpha': '[0.00, 10.00] (Float)',
        'reg_lambda': '[0.00, 10.00] (Float)',
    }

    print("\n[4/4] HASIL OPTIMASI & PARAMETER TERBAIK:", flush=True)
    print(f"  - Skor Terbaik (PR-AUC)    : {trial.value:.4f} (Tercapai pada Trial #{trial.number + 1})", flush=True)
    print("  " + "-" * 74, flush=True)
    print(f"  {'Nama Parameter':<20} {'Nilai Terpilih':<18} {'Rentang Pencarian (Search Space)':<32}", flush=True)
    print("  " + "-" * 74, flush=True)
    for key, space in search_spaces.items():
        val = trial.params.get(key, "-")
        if isinstance(val, float):
            val_str = f"{val:.4f}"
        else:
            val_str = str(val)
        print(f"  {key:<20} {val_str:<18} {space:<32}", flush=True)
    print("  " + "-" * 74, flush=True)

    # Save best parameters to JSON
    model_dir = os.path.join(_PROJECT_ROOT, "evaluation", "models", "xgboost")
    os.makedirs(model_dir, exist_ok=True)

    best_params_path = os.path.join(model_dir, "best_params.json")
    with open(best_params_path, "w") as f:
        json.dump(trial.params, f, indent=4)

    rel_best_params = os.path.relpath(best_params_path, _PROJECT_ROOT).replace('\\', '/')
    print("\n[STATUS PENYIMPANAN ARTEFAK & LANGKAH SELANJUTNYA]", flush=True)
    print(f"  - File Konfigurasi : {rel_best_params}", flush=True)
    print("  - Rekomendasi      : Jalankan skrip berikut untuk melatih model final:", flush=True)
    print("                       .venv\\Scripts\\python.exe src/stage4_modeling/xgboost/06_train_xgb.py", flush=True)
    print("=" * 78 + "\n", flush=True)


if __name__ == "__main__":
    main()
