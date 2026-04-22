import pandas as pd
import numpy as np
import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, cross_val_score
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

try:
    from imblearn.combine import SMOTETomek
    HAS_IMBLEARN = True
except ImportError:
    HAS_IMBLEARN = False
    print("Warning: imbalanced-learn is not installed. SMOTE will not be available. Use `pip install imbalanced-learn`.")

class DamageDetector:
    def __init__(self, model_path='road_damage_model.pkl'):
        # Menentukan path absolut jika dipanggil dari direktori lain
        _dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
        self.model_path = os.path.join(_dir, model_path)
        self.model = None
        
        # Sesuai dengan hasil ekstraksi fitur jendela terbaru (windows_features.csv + Gyro)
        self.feature_columns = [
            'mag_mean', 'mag_std', 'mag_max', 'mag_rms', 'mag_jrk', 'speed_mean',
            'gyro_mag_mean', 'gyro_mag_max', 'gyro_mag_jrk',
            'gyro_x_std', 'gyro_y_std', 'gyro_z_std'
        ]

    def load_training_data(self, data_path):
        """
        Loads the labeled training data.
        Cleans any rows with missing features or missing labels.
        """
        try:
            df = pd.read_csv(data_path)
            
            # Pengecekan fitur
            missing_cols = [col for col in self.feature_columns if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing feature columns in training data: {missing_cols}")
                
            if 'label' not in df.columns:
                raise ValueError("Missing 'label' column in training data.")
            
            # Hapus baris yang kebetulan label-nya kosong
            df_clean = df.dropna(subset=['label']).copy()
            
            # Handle nilai fitur kosong (misal speed hilang sewaktu hilang sinyal GPS)
            df_clean[self.feature_columns] = df_clean[self.feature_columns].fillna(df_clean[self.feature_columns].median())
            
            print(f"Data successfully loaded. Total ready shape: {df_clean.shape}")
            return df_clean
        except Exception as e:
            print(f"Error loading training data: {e}")
            return None

    def train(self, X, y, groups=None, test_size=0.2, random_state=42):
        """
        Train the Random Forest model using Trip-based Split.
        
        Metode Trip-based Split menjamin bahwa seluruh window dari satu trip
        masuk ke Train ATAU Test secara utuh, tidak pernah tercampur.
        Ini mencegah data leakage akibat korelasi temporal antar window
        yang berdekatan dalam satu perjalanan.
        
        :param X: Feature matrix.
        :param y: Target labels.
        :param groups: Series/array berisi trip_id untuk setiap baris.
        """
        print("Mempersiapkan data pelatihan (Trip-based Split)...")
        
        if groups is not None:
            # === TRIP-BASED SPLIT ===
            # GroupShuffleSplit memastikan semua window dari 1 trip
            # masuk ke Train atau Test secara utuh, TIDAK dicampur.
            gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
            train_idx, test_idx = next(gss.split(X, y, groups))
            
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            # Informasi pembagian trip
            train_trips = groups.iloc[train_idx].nunique()
            test_trips = groups.iloc[test_idx].nunique()
            print(f"  Trip untuk Training : {train_trips} trip ({len(X_train)} windows)")
            print(f"  Trip untuk Testing  : {test_trips} trip ({len(X_test)} windows)")
            print(f"  Total Trip          : {groups.nunique()} trip")
        else:
            # Fallback ke split biasa jika trip_id tidak tersedia
            from sklearn.model_selection import train_test_split
            print("  [INFO] trip_id tidak diberikan, menggunakan random split biasa.")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, stratify=y, random_state=random_state
            )
        
        # Menerapkan SMOTETomek (Synthetic Minority + Cleaning)
        if HAS_IMBLEARN:
            print("Menerapkan teknik SMOTE untuk menyeimbangkan kelas minoritas...")
            smote = SMOTETomek(random_state=random_state)
            X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
        else:
            X_train_res, y_train_res = X_train, y_train

        # Konfigurasi Model (Produksi)
        self.model = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_leaf=5,
            class_weight='balanced',
            random_state=random_state,
            n_jobs=-1
        )
        
        print("\nProses Melatih (Training) Model Random Forest dimulai...")
        self.model.fit(X_train_res, y_train_res)
        
        print("\n--- HASIL EVALUASI (TRIP-BASED SPLIT) ---")
        y_pred = self.model.predict(X_test)
        print(f"Akurasi Keseluruhan: {accuracy_score(y_test, y_pred)*100:.2f}%")
        print("\nLaporan Detail Presisi/Recall:")
        print(classification_report(y_test, y_pred))
        
        # Simpan index test untuk keperluan visualisasi confusion matrix di notebook
        self._last_test_idx = test_idx if groups is not None else None
        self._last_X_test = X_test
        self._last_y_test = y_test
        
        return self.model
        
    def cross_validate(self, X, y, groups=None, n_splits=5):
        """
        Assess Model quality using Group K-Fold Cross-Validation.
        Setiap fold berisi trip-trip utuh yang berbeda, sehingga
        model benar-benar diuji terhadap perjalanan yang belum pernah dilihat.
        
        :param groups: Series/array berisi trip_id.
        """
        if not self.model:
            print("Tolong train terlebih dahulu sebelum validasi.")
            return
        
        if groups is not None:
            unique_trips = groups.nunique()
            actual_splits = min(n_splits, unique_trips)
            print(f"\n--- HASIL UJI SILANG ({actual_splits}-Fold Group/Trip-based) ---")
            print(f"    Total trip unik: {unique_trips}, dibagi ke {actual_splits} fold.")
            gkf = GroupKFold(n_splits=actual_splits)
            scores = cross_val_score(self.model, X, y, cv=gkf, groups=groups, scoring='f1_weighted', n_jobs=-1)
        else:
            from sklearn.model_selection import StratifiedKFold
            print(f"\n--- HASIL UJI SILANG ({n_splits}-Fold Stratified, tanpa trip grouping) ---")
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            scores = cross_val_score(self.model, X, y, cv=skf, scoring='f1_weighted', n_jobs=-1)
            
        print(f"  F1-Score tiap fold: {scores}")
        print(f"  F1-Score Rata-rata: {scores.mean():.4f} (+/- {scores.std():.4f})")
        
        if scores.mean() < 0.7:
            print("  ⚠️  F1-Score < 0.70. Model mungkin overfitting pada trip tertentu.")
        else:
            print("  ✅ Model memiliki generalisasi yang baik antar trip!")

    def save_model(self):
        """Save the trained model to disk."""
        if self.model:
            joblib.dump(self.model, self.model_path)
            print(f"\n[SUCCESS] Model produksi tersimpan di: {self.model_path}")
        else:
            print("No model to save. Train first.")

    def load_model(self):
        """Load a trained model from disk."""
        if os.path.exists(self.model_path):
            self.model = joblib.load(self.model_path)
            print(f"Model berhasil di-load dari {self.model_path}")
        else:
            print(f"File model tidak ditemukan di {self.model_path}")

    def predict(self, features_df):
        """
        Predict class and confidence. Returns single labels or bulk predictions.
        """
        if not self.model:
            self.load_model()
            if not self.model:
                raise Exception("Fatal: Tidak ada model yang diload/dilatih!")

        # Pastikan fiturnya urut dan lengkap
        missing = [c for c in self.feature_columns if c not in features_df.columns]
        if missing:
            raise ValueError(f"Data tidak punya fitur: {missing}")
            
        X = features_df[self.feature_columns].fillna(features_df[self.feature_columns].median())
        
        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)
        confidence = np.max(probabilities, axis=1)
        
        results = pd.DataFrame({
            'prediction': predictions,
            'confidence': confidence
        }, index=features_df.index)
        
        return results

if __name__ == "__main__":
    # Test script jika dipanggil langsung
    detector = DamageDetector()
    print("Class siap digunakan.")
