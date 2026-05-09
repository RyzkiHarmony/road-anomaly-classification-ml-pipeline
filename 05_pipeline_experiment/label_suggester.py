# label_suggester.py
# Semi-supervised labeling assist for motorcycle road-anomaly events.
#
# PURPOSE:
#   - Suggest labels based on sensor features (rule-based + ML hybrid)
#   - Never overwrite manual labels — suggestions only
#   - Provide confidence, reason, and review flags
#
# TAXONOMY (3-class):
#   - "Non-Event"  : semua yang bukan event diskrit (rough road, engine vibration,
#                    maneuver, noise, crack, normal background)
#   - "Pothole"    : lubang jalan
#   - "Speed Bump" : polisi tidur

from __future__ import annotations

import os
import numpy as np
import pandas as pd
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import LabelEncoder
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False


# ---------- LABEL TAXONOMY ----------
# Single source of truth for the 3-class classification schema.
# All rule-based and ML-based suggestions MUST resolve to one of these.
VALID_LABELS = ("Non-Event", "Pothole", "Speed Bump")

# Mapping from internal diagnostic raw labels to the official taxonomy.
# Raw labels preserve detection nuance for debugging and analysis;
# the final suggested_label is always one of VALID_LABELS.
_RAW_TO_FINAL = {
    "Non-Event":   "Non-Event",
    "Pothole":     "Pothole",
    "Speed Bump":  "Speed Bump",
    # Diagnostic sub-categories → collapsed into Non-Event
    "Maneuver":    "Non-Event",
    "Rough Road":  "Non-Event",
    "Normal":      "Non-Event",
    "Crack":       "Non-Event",
}

# ---------- ML FEATURES ----------
ML_FEATURES = [
    "peak_vertical_g", "num_peaks_accel", "num_peaks_gyro",
    "asymmetry_score", "local_duration", "vertical_energy", "gyro_energy",
    "speed_mean", "vert_jrk", "fft_high_low_ratio", "max_jerk",
    "gyro_pitch_energy", "gyro_roll_energy", "gyro_pitch_roll_ratio",
    "kurtosis", "skewness", "zcr"
]

# ---------- SUGGESTION SCHEMA ----------
SUGGESTION_COLS = [
    "suggested_label",
    "suggested_raw_label",
    "suggested_kind",        # event / condition_like / ambiguous
    "suggestion_confidence",
    "suggestion_rule",
    "suggestion_reason",
    "needs_review",
]


# ---------- ML MODEL TRAINING ----------

def train_ml_models(df_candidates, gt_path):
    """Train RandomForest and XGBoost on existing ground truth."""
    models = {
        "rf": None, "rf_classes": None,
        "xgb": None, "xgb_le": None
    }
    
    if not os.path.exists(gt_path):
        return models

    try:
        gt = pd.read_csv(gt_path)
        if "event_id" not in gt.columns or "label" not in gt.columns:
            return models

        merged = pd.merge(df_candidates, gt, on="event_id", suffixes=("", "_gt"))
        merged = merged.dropna(subset=ML_FEATURES + ["label"])

        # Hanya latih jika minimal ada 2 kelas dan > 5 data
        if len(merged) < 5 or merged["label"].nunique() < 2:
            return models

        X = merged[ML_FEATURES].values
        y = merged["label"].values

        if HAS_SKLEARN:
            try:
                rf = RandomForestClassifier(n_estimators=50, random_state=42, max_depth=5)
                rf.fit(X, y)
                models["rf"] = rf
                models["rf_classes"] = list(rf.classes_)
            except Exception as e:
                print(f"  [WARN] RF gagal dilatih: {e}")

        if HAS_XGB and HAS_SKLEARN:
            try:
                le = LabelEncoder()
                y_enc = le.fit_transform(y)
                xgb = XGBClassifier(n_estimators=50, random_state=42, max_depth=3, eval_metric='mlogloss')
                xgb.fit(X, y_enc)
                models["xgb"] = xgb
                models["xgb_le"] = le
            except Exception as e:
                print(f"  [WARN] XGBoost gagal dilatih: {e}")

    except Exception as e:
        print(f"  [WARN] ML Suggester gagal dilatih: {e}")

    return models


# ---------- HELPERS ----------

def _f(row, key, default=np.nan):
    try:
        v = row.get(key, default)
        return float(v) if v is not None and pd.notna(v) else default
    except Exception:
        return default


def _in_range(x, lo, hi):
    return pd.notna(x) and lo <= x <= hi


# ---------- CORE SUGGESTION LOGIC ----------

def suggest_event_label(row: pd.Series, models=None) -> pd.Series:
    """
    Suggest a label for a single candidate event row.

    Output taxonomy (3-class):
      - "Non-Event"  : semua yang bukan event diskrit
      - "Pothole"    : lubang jalan
      - "Speed Bump" : polisi tidur

    Pendekatan Hybrid:
    1. Soft-Voting Ensemble (Rata-rata probabilitas RF & XGBoost).
    2. Fallback ke Rule-based Heuristics jika confidence ML rendah (< 65%)
       atau model belum tersedia.
    """
    if models is None:
        models = {}
        
    rf_model = models.get("rf")
    rf_classes = models.get("rf_classes")
    xgb_model = models.get("xgb")
    xgb_le = models.get("xgb_le")

    # Ambil fitur untuk heuristics
    a = abs(_f(row, "peak_vertical_g"))  # use absolute vertical G for old magnitude thresholds
    g = _f(row, "peak_gyro_mag")
    npa = int(_f(row, "num_peaks_accel", 0))
    npg = int(_f(row, "num_peaks_gyro", 0))
    asym = _f(row, "asymmetry_score")
    dur = _f(row, "local_duration")
    fft_ratio = _f(row, "fft_high_low_ratio", 0.0)
    max_jrk = _f(row, "max_jerk", 0.0)
    p_mag = _f(row, "peak_mag", 0.0)
    
    # S-Tier Features
    pitch_e = _f(row, "gyro_pitch_energy", 0.0)
    roll_e = _f(row, "gyro_roll_energy", 0.0)
    pr_ratio = _f(row, "gyro_pitch_roll_ratio", 0.0)
    skew = _f(row, "skewness", 0.0)
    kur = _f(row, "kurtosis", 0.0)
    zcr = _f(row, "zcr", 0.0)

    best_confidence = 0.0
    best_label = "Non-Event"
    best_model_name = "NONE"

    # --- 1. PREDIKSI ML (SOFT-VOTING ENSEMBLE) ---
    try:
        x_vals = [row.get(f, 0.0) for f in ML_FEATURES]
        x_vals = [float(x) if pd.notna(x) else 0.0 for x in x_vals]
        X_pred = np.array([x_vals])

        probs_list = []
        used_models = []

        if rf_model is not None and rf_classes is not None:
            rf_probs = rf_model.predict_proba(X_pred)[0]
            probs_list.append({"probs": rf_probs, "classes": rf_classes})
            used_models.append("RF")

        if xgb_model is not None and xgb_le is not None:
            xgb_probs = xgb_model.predict_proba(X_pred)[0]
            xgb_classes = xgb_le.inverse_transform(np.arange(len(xgb_probs)))
            probs_list.append({"probs": xgb_probs, "classes": list(xgb_classes)})
            used_models.append("XGB")

        if probs_list:
            # Hitung rata-rata probabilitas untuk setiap kelas yang dikenal
            all_known_classes = set()
            for p in probs_list:
                all_known_classes.update(p["classes"])
            
            avg_probs = {}
            for cls in all_known_classes:
                sum_prob = 0.0
                for p in probs_list:
                    if cls in p["classes"]:
                        idx = p["classes"].index(cls)
                        sum_prob += p["probs"][idx]
                avg_probs[cls] = sum_prob / len(probs_list)

            # Cari kelas dengan rata-rata probabilitas tertinggi
            best_label = max(avg_probs, key=avg_probs.get)
            best_confidence = avg_probs[best_label]
            best_model_name = "Ensemble(" + "+".join(used_models) + ")"

    except Exception:
        pass

    # Jika ML sangat yakin, langsung kembalikan
    if best_model_name != "NONE" and best_confidence >= 0.65:
        kind = "event" if best_label in ("Pothole", "Speed Bump") else "condition_like"
        rule = f"ML_{best_model_name}"
        reason = f"Dipelajari dari data (Confidence: {best_confidence*100:.0f}%)"
        
        suggested_label = _RAW_TO_FINAL.get(best_label, "Non-Event")
        return pd.Series({
            "suggested_label": suggested_label,
            "suggested_raw_label": best_label,
            "suggested_kind": kind,
            "suggestion_confidence": float(best_confidence),
            "suggestion_rule": rule,
            "suggestion_reason": reason,
            "needs_review": bool(best_confidence < 0.80),
        })

    # --- 2. FALLBACK HEURISTICS (RULES) ---
    raw_label = "Non-Event"
    kind = "ambiguous"
    rule = "R9"
    reason = "fallback"
    confidence = 0.40
    needs_review = True

    # R10 — Extreme Shock: Amplitudo raksasa (> 45 m/s²)
    if (p_mag >= 45):
        # Speed Bump WAJIB punya energi pitch yang dominan
        if pr_ratio > 1.8 and pitch_e > 40:
            raw_label = "Speed Bump"
            reason = f"Extreme shock ({p_mag:.1f} m/s²) + High Pitch/Roll Ratio ({pr_ratio:.1f}) → Speed Bump"
        else:
            raw_label = "Pothole"
            reason = f"Extreme shock ({p_mag:.1f} m/s²) + Low Pitch/Roll Ratio ({pr_ratio:.1f}) → Pothole"
        kind = "event"
        rule = "R10"
        confidence = 0.98
        needs_review = False

    # R5 — Maneuver: gyro dominan + asimetri tinggi
    elif (npg >= 3) and (asym > 0.60) and (a < 12):
        raw_label = "Maneuver"
        kind = "condition_like"
        rule = "R5"
        reason = "gyro dominan + asimetri tinggi + low accel → maneuver (Non-Event)"
        confidence = 0.88
        needs_review = False

    # R1 — Pothole: impulse tajam + high freq energy + negative skew
    elif (a >= 15) and (fft_ratio >= 10.0 or skew < -0.5):
        raw_label = "Pothole"
        kind = "event"
        rule = "R1"
        reason = f"impulse tajam + FFT High ({fft_ratio:.1f}) + Skewness ({skew:.2f}) → Pothole"
        confidence = 0.95
        needs_review = False

    # R3 — Speed Bump: low freq energy + pitch energy dominant
    elif (a >= 12) and (fft_ratio < 5.0) and (pr_ratio > 1.5):
        raw_label = "Speed Bump"
        kind = "event"
        rule = "R3"
        reason = f"low frequency energy (FFT < 5) + Pitch Dominant (P/R Ratio: {pr_ratio:.1f})"
        confidence = 0.92
        needs_review = False

    # R7 — Rough Road: Guncangan kontinu tapi tidak ekstrem → Non-Event
    elif (a >= 10) and (npa >= 3) and (dur > 0.4):
        raw_label = "Rough Road"
        kind = "condition_like"
        rule = "R7"
        reason = "struktur kontinu panjang + low-mid magnitude → Rough Road (Non-Event)"
        confidence = 0.75
        needs_review = True

    # R6 — Normal / background vibration
    elif (a < 6) and (npa <= 1):
        raw_label = "Normal"
        kind = "condition_like"
        rule = "R6"
        reason = "amplitudo rendah + tidak ada struktur event → Non-Event"
        confidence = 0.92
        needs_review = False
        
    # R9 - Fallback ML Low Confidence
    elif best_model_name != "NONE":
        raw_label = best_label
        kind = "event" if best_label in ("Pothole", "Speed Bump") else "condition_like"
        rule = f"ML_{best_model_name}_LOW_CONF"
        reason = f"ML ragu-ragu (Conf: {best_confidence*100:.0f}%), pakai tebakan terbaik ML"
        confidence = float(best_confidence)
        needs_review = True

    suggested_label = _RAW_TO_FINAL.get(raw_label, "Non-Event")

    return pd.Series({
        "suggested_label": suggested_label,
        "suggested_raw_label": raw_label,
        "suggested_kind": kind,
        "suggestion_confidence": float(confidence),
        "suggestion_rule": rule,
        "suggestion_reason": reason,
        "needs_review": bool(needs_review),
    })


# ---------- PUBLIC API ----------

def apply_label_suggestions(df: pd.DataFrame, gt_path: str = None) -> pd.DataFrame:
    """
    Tambahkan kolom sugesti ke DataFrame event menggunakan ML + rules.

    Output taxonomy: Non-Event, Pothole, Speed Bump.
    """
    if df is None or df.empty:
        out = df.copy() if df is not None else pd.DataFrame()
        for c in SUGGESTION_COLS:
            out[c] = []
        return out

    # Default gt path relatif ke script ini
    if gt_path is None:
        _SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
        gt_path = os.path.join(_SCRIPT_DIR, "out", "ground_truth_labels.csv")

    out = df.copy()

    models = train_ml_models(out, gt_path)
    
    active_models = []
    if models.get("rf") is not None: active_models.append("RandomForest")
    if models.get("xgb") is not None: active_models.append("XGBoost")
    
    if active_models:
        print(f"  [INFO] ML Suggester diaktifkan. Model aktif: {', '.join(active_models)}")

    sugg = out.apply(lambda r: suggest_event_label(r, models=models), axis=1)
    out = pd.concat([out, sugg], axis=1)

    out["needs_review"] = out["needs_review"].astype(bool)
    return out


def save_label_suggestions(df: pd.DataFrame, out_path: str) -> None:
    """Simpan file sugesti terpisah."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cols = [c for c in df.columns if c in (
        "event_id", "trip_id", "time_s", "lat", "lon",
        "peak_vertical", "peak_vertical_g", "peak_gyro_mag",
        "speed_mean", "event_duration", "vert_jrk",
        "num_peaks_accel", "num_peaks_gyro",
        "peak_interval_mean", "peak_interval_std",
        "asymmetry_score", "vertical_energy", "gyro_energy",
        "accel_to_gyro_ratio", "local_duration",
        "score", "priority", "level",
        *SUGGESTION_COLS
    )]

    df[cols].to_csv(out_path, index=False)