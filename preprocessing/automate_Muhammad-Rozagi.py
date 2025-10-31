

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from pathlib import Path
import sys

# ======================
# 1. Setup Konstanta & Path
# ======================
RAW_DATA_PATH = Path("heart_raw/heart.csv") 
OUTPUT_FOLDER = Path("preprocessing/heart_preprocessing")
OUTPUT_FILE = OUTPUT_FOLDER / "heart_preprocessed.csv"

# (Konstanta rename & daftar kolom a)
COLUMN_RENAME_MAP = {
    "age": "Age", "sex": "Sex", "cp": "ChestPain",
    "trestbps": "RestingBloodPressure", "chol": "Cholesterol",
    "fbs": "FastingBloodSugar", "restecg": "RestingECG",
    "thalach": "MaxHeartRate", "exang": "ExcerciseAngina",
    "oldpeak": "OldPeak", "slope": "STSlope",
    "ca": "nMajorVessels", "thal": "Thalium", "target": "Status"
}
CATEGORICAL_COLS = [
    'Sex', 'ChestPain', 'FastingBloodSugar', 'RestingECG',
    'ExcerciseAngina', 'STSlope', 'Thalium'
]
NUMERIC_COLS = [
    'Age', 'RestingBloodPressure', 'Cholesterol',
    'MaxHeartRate', 'OldPeak', 'nMajorVessels'
]
COLS_TO_CAP = [
    'RestingBloodPressure', 'Cholesterol', 'MaxHeartRate', 'OldPeak'
]

# ======================
# 2. Fungsi Preprocessing Utama
# ======================

def run_preprocessing():
    """
    Memuat data, membersihkan (duplikat, outlier), lalu
    membagi data (split) SEBELUM menerapkan scaling/encoding
    untuk mencegah data leakage.
    """
    print("Memulai skrip preprocessing otomatis (v2)...")

    # --- 1. Load, Rename, Drop Duplikat, Cap Outlier ---
    try:
        df_raw = pd.read_csv(RAW_DATA_PATH)
    except FileNotFoundError:
        print(f"ERROR: File data mentah tidak ditemukan di {RAW_DATA_PATH}")
        sys.exit(1)

    df_model = df_raw.rename(columns=COLUMN_RENAME_MAP)
    
    len_before = len(df_model)
    df_model = df_model.drop_duplicates()
    print(f"Menghapus {len_before - len(df_model)} baris duplikat.")

    print("Menangani outlier dengan Capping IQR...")
    for col in COLS_TO_CAP:
        if col in df_model.columns:
            Q1 = df_model[col].quantile(0.25)
            Q3 = df_model[col].quantile(0.75)
            IQR = Q3 - Q1
            batas_bawah = Q1 - (1.5 * IQR)
            batas_atas = Q3 + (1.5 * IQR)
            df_model[col] = df_model[col].clip(lower=batas_bawah, upper=batas_atas)
    
    # --- 2. Pisahkan Fitur (X) dan Target (y) ---
    X = df_model.drop(columns=['Status'])
    y = df_model['Status']

    # --- 3. Train-Test Split (SEBELUM PREPROCESSING) ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    print("Data berhasil di-split (70% train, 30% test).")

    # --- 4. One-Hot Encoding ---
    # (pd.get_dummies tidak rawan leakage, tapi lebih baik konsisten)
    # Kita 'fit' di semua data X agar kolomnya konsisten
    X_cat_full = pd.get_dummies(
        X[CATEGORICAL_COLS].astype(str),
        columns=CATEGORICAL_COLS,
        drop_first=True
    )
    # Lalu pisahkan lagi berdasarkan index
    X_train_cat = X_cat_full.loc[X_train.index]
    X_test_cat = X_cat_full.loc[X_test.index]

    # --- 5. Scaling Fitur Numerik (CARA AMAN) ---
    scaler = StandardScaler()
    
    # 'fit' HANYA di X_train
    X_train_num_scaled = scaler.fit_transform(X_train[NUMERIC_COLS])
    # 'transform' di X_test (pakai mean/std dari X_train)
    X_test_num_scaled = scaler.transform(X_test[NUMERIC_COLS])

    # Konversi kembali ke DataFrame
    num_scaled_cols = [f"{col}_scaled" for col in NUMERIC_COLS]
    X_train_num_df = pd.DataFrame(X_train_num_scaled, columns=num_scaled_cols, index=X_train.index)
    X_test_num_df = pd.DataFrame(X_test_num_scaled, columns=num_scaled_cols, index=X_test.index)
    
    # --- 6. Gabungkan Kembali ---
    X_train_processed = pd.concat([X_train_num_df, X_train_cat], axis=1)
    X_test_processed = pd.concat([X_test_num_df, X_test_cat], axis=1)

    # Gabungkan dengan target
    train_final_df = pd.concat([X_train_processed, y_train.reset_index(drop=True)], axis=1)
    test_final_df = pd.concat([X_test_processed, y_test.reset_index(drop=True)], axis=1)
    
    # Gabungkan semua data untuk disimpan (sesuai permintaan tugas)
    final_df = pd.concat([train_final_df, test_final_df]).reset_index(drop=True)
    
    # --- 7. Simpan Dataset ---
    print("Menyimpan dataset yang sudah diproses...")
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSukses! Data telah diproses dan disimpan di:\n{OUTPUT_FILE}")

# ======================
# 3. Eksekusi Skrip
# ======================
if __name__ == "__main__":
    run_preprocessing()
