import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from pathlib import Path 
import sys

# ======================
# 1. Setup Konstanta & Path
# ======================
# Sdata mentah ada di folder 'heart_raw'
RAW_DATA_PATH = Path("heart_raw/heart.csv") 

#  output disimpan di 'heart_preprocessing'
OUTPUT_FOLDER = Path("preprocessing/heart_preprocessing")
OUTPUT_FILE = OUTPUT_FOLDER / "heart_preprocessed.csv"

# Konstanta rename (diambil dari notebook)
COLUMN_RENAME_MAP = {
    "age": "Age", "sex": "Sex", "cp": "ChestPain",
    "trestbps": "RestingBloodPressure", "chol": "Cholesterol",
    "fbs": "FastingBloodSugar", "restecg": "RestingECG",
    "thalach": "MaxHeartRate", "exang": "ExcerciseAngina",
    "oldpeak": "OldPeak", "slope": "STSlope",
    "ca": "nMajorVessels", "thal": "Thalium", "target": "Status"
}

# Konstanta kolom (diambil dari notebook)
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
    Fungsi utama untuk memuat, memproses, dan menyimpan dataset.
    """
    print("Memulai skrip preprocessing otomatis...")

    # --- 1. Load Data ---
    try:
        df_raw = pd.read_csv(RAW_DATA_PATH)
        print(f"Data mentah {RAW_DATA_PATH} berhasil dimuat.")
    except FileNotFoundError:
        print(f"ERROR: File data mentah tidak ditemukan di {RAW_DATA_PATH}")
        print("Pastikan file 'heart.csv' ada di folder 'namadataset_raw/'")
        sys.exit(1) # Keluar dari skrip jika file tidak ada

    # --- 2. Rename Kolom ---
    df_model = df_raw.rename(columns=COLUMN_RENAME_MAP)

    # --- 3. Hapus Duplikat ---
    len_before = len(df_model)
    df_model = df_model.drop_duplicates()
    len_after = len(df_model)
    print(f"Menghapus {len_before - len_after} baris duplikat.")

    # --- 4. Penanganan Outlier ---
    print("Menangani outlier dengan Capping IQR...")
    for col in COLS_TO_CAP:
        if col in df_model.columns:
            Q1 = df_model[col].quantile(0.25)
            Q3 = df_model[col].quantile(0.75)
            IQR = Q3 - Q1
            batas_bawah = Q1 - (1.5 * IQR)
            batas_atas = Q3 + (1.5 * IQR)
            df_model[col] = df_model[col].clip(lower=batas_bawah, upper=batas_atas)
    print("Penanganan outlier selesai.")

    # --- 5. Pisahkan Fitur (X) dan Target (y) ---
    # Dijalankan SETELAH data bersih (duplikat & outlier)
    X = df_model.drop(columns=['Status'])
    y = df_model['Status'].reset_index(drop=True) 

    # --- 6. One-Hot Encoding Fitur Kategorikal ---
    print("Melakukan One-Hot Encoding...")
    X_cat = pd.get_dummies(
        X[CATEGORICAL_COLS].astype(str),
        columns=CATEGORICAL_COLS,
        drop_first=True
    ).reset_index(drop=True)

    # --- 7. Scaling Fitur Numerik ---
    print("Melakukan Standard Scaling...")
    scaler = StandardScaler()
    X_num_scaled = pd.DataFrame(
        scaler.fit_transform(X[NUMERIC_COLS]),
        columns=[f"{col}_scaled" for col in NUMERIC_COLS]
    ).reset_index(drop=True)

    # --- 8. Gabungkan Kembali ---
    print("Menggabungkan semua fitur yang sudah diproses...")
    X_processed = pd.concat([X_num_scaled, X_cat], axis=1)
    
    # Gabungkan dengan target ke satu dataframe final
    final_df = pd.concat([X_processed, y], axis=1)

    # --- 9. Simpan Dataset ---
    print("Menyimpan dataset yang sudah diproses...")
    # Pastikan folder output ada
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    
    final_df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSukses! Data telah diproses dan disimpan di:\n{OUTPUT_FILE}")

# ======================
# 3. Eksekusi Skrip
# ======================
if __name__ == "__main__":
    # Ini membuat skrip bisa dijalankan langsung dari terminal/Actions
    run_preprocessing()
