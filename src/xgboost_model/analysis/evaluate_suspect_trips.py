import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import joblib
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings("ignore")

from config import OUT_FOLDER, BEST_FEATURES

def evaluate_trip(df, trip_id, model, le, feature_cols):
    trip_df = df[df['trip_id'] == trip_id]
    if trip_df.empty:
        print(f"Trip {trip_id} tidak ditemukan atau kosong.")
        return

    X = trip_df[feature_cols].values
    y_true = trip_df['label'].values
    
    y_pred_enc = model.predict(X)
    y_pred = le.inverse_transform(y_pred_enc)
    
    print(f"\n{'='*50}")
    print(f"EVALUASI TRIP: {trip_id}")
    print(f"Jumlah Sampel: {len(trip_df)}")
    print(f"{'='*50}")
    
    # Check if there are any non-nan labels
    if len(y_true) > 0:
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, zero_division=0))
        
        print("\nConfusion Matrix:")
        labels = le.classes_
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        cm_df = pd.DataFrame(cm, index=[f"True {l}" for l in labels], columns=[f"Pred {l}" for l in labels])
        print(cm_df)
    else:
        print("Tidak ada label valid untuk trip ini.")

def main():
    data_path = os.path.join(OUT_FOLDER, "manual_labeled_windows.csv")
    df = pd.read_csv(data_path)
    df = df.dropna(subset=['label'])
    
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    model_path = os.path.join(base_dir, "models", "best_model.pkl")
    le_path = os.path.join(base_dir, "models", "label_encoder.pkl")
    
    model = joblib.load(model_path)
    le = joblib.load(le_path)
    
    feature_cols = [c for c in df.columns if c in BEST_FEATURES]
    df = df.dropna(subset=feature_cols)
    
    suspect_trips = [
        "bf026662-aa33-4c79-b34d-296f7893c390",
        "fe628252-e600-47d4-875d-7c6f005a735e"
    ]
    
    for trip_id in suspect_trips:
        evaluate_trip(df, trip_id, model, le, feature_cols)

if __name__ == "__main__":
    main()
