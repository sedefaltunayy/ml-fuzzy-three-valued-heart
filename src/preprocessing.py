"""
preprocessing.py
----------------
Ham kardiyovasküler veriyi temizleyen ve dönüştüren modül.

Adımlar:
  1. CSV yükleme (otomatik delimiter detection: ; önce, , sonra)
  2. Sütun adlarını standartlaştır
  3. age (gün) → yıla çevir  (age / 365.25)
  4. BMI hesapla            (weight / (height/100)^2)
  5. Tıbbi outlier sınırları uygula
  6. Eksik değer yönetimi
  7. Gereksiz / duplike satır temizliği
"""

import os
import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# Tıbbi outlier sınırları
# ---------------------------------------------------------------------------
OUTLIER_BOUNDS = {
    "ap_hi":   (80, 250),    # sistolik kan basıncı mmHg
    "ap_lo":   (50, 150),    # diastolik kan basıncı mmHg
    "height":  (100, 250),   # boy cm
    "weight":  (30, 250),    # kilo kg
    "bmi":     (10, 70),     # BMI
}


# ---------------------------------------------------------------------------
# CSV Yükleme
# ---------------------------------------------------------------------------

def load_csv(filepath: str) -> pd.DataFrame:
    """
    CSV dosyasını otomatik delimiter detection ile yükler.
    Önce ';' denenir, ardından ',' denenir.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dosya bulunamadı: {filepath}")

    for sep in [";", ","]:
        try:
            df = pd.read_csv(filepath, sep=sep)
            # Anlamlı sütun sayısı kontrolü (en az 5)
            if df.shape[1] >= 5:
                return df
        except Exception:
            continue

    raise ValueError(f"CSV dosyası okunamadı: {filepath}")


# ---------------------------------------------------------------------------
# Sütun adı standardizasyonu
# ---------------------------------------------------------------------------

COLUMN_RENAME_MAP = {
    # Kaggle Cardiovascular Disease dataset varyasyonları
    "id":           "record_id",
    "age":          "age",
    "gender":       "gender",
    "height":       "height",
    "weight":       "weight",
    "ap_hi":        "ap_hi",
    "ap_lo":        "ap_lo",
    "cholesterol":  "cholesterol",
    "gluc":         "gluc",
    "smoke":        "smoke",
    "alco":         "alco",
    "active":       "active",
    "cardio":       "cardio",
}


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sütun adlarını küçük harfe çevirir, boşlukları alt çizgiyle değiştirir.
    Bilinen yeniden adlandırmaları uygular.
    """
    df = df.copy()
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # 'record_id' veya 'id' sütunu varsa bırak
    return df


# ---------------------------------------------------------------------------
# Yaş dönüşümü
# ---------------------------------------------------------------------------

def convert_age(df: pd.DataFrame) -> pd.DataFrame:
    """
    'age' sütunu gün cinsinden ise yıla çevirir.
    Eğer değerler > 200 ise gün olarak kabul edilir.
    """
    df = df.copy()
    if "age" in df.columns:
        if df["age"].median() > 200:
            # Gün → Yıl
            df["age_years"] = (df["age"] / 365.25).round(2)
        else:
            # Zaten yıl cinsinden
            df["age_years"] = df["age"].round(2)
    return df


# ---------------------------------------------------------------------------
# BMI hesaplama
# ---------------------------------------------------------------------------

def calculate_bmi(df: pd.DataFrame) -> pd.DataFrame:
    """
    BMI = weight (kg) / (height (cm) / 100)^2
    Sıfır bölünme ve geçersiz değerleri NaN ile işaretler.
    """
    df = df.copy()
    if "height" in df.columns and "weight" in df.columns:
        height_m = df["height"] / 100.0
        df["bmi"] = (df["weight"] / (height_m ** 2)).round(2)
        # Geçersiz BMI'yi NaN yap
        df.loc[height_m <= 0, "bmi"] = np.nan
    return df


# ---------------------------------------------------------------------------
# Outlier temizliği
# ---------------------------------------------------------------------------

def remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    OUTLIER_BOUNDS sözlüğündeki tıbbi sınırlara göre satırları filtreler.
    Sınır dışı kalan satırlar tamamen kaldırılır.
    """
    df = df.copy()
    initial_len = len(df)
    for col, (low, high) in OUTLIER_BOUNDS.items():
        if col in df.columns:
            df = df[(df[col] >= low) & (df[col] <= high)]

    removed = initial_len - len(df)
    if removed > 0:
        print(f"[Preprocessing] Outlier temizliği: {removed} satır kaldırıldı.")
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Eksik değer yönetimi
# ---------------------------------------------------------------------------

def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sayısal sütunlar için medyan, kategorik sütunlar için mod ile doldurur.
    Doldurma sonrası hâlâ NaN olan satırlar kaldırılır.
    """
    df = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns

    for col in numeric_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())

    for col in categorical_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].mode()[0])

    df = df.dropna()
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Duplike temizliği
# ---------------------------------------------------------------------------

def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Tam duplike satırları kaldırır."""
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    after = len(df)
    if before - after > 0:
        print(f"[Preprocessing] {before - after} duplike satır kaldırıldı.")
    return df


# ---------------------------------------------------------------------------
# Ana pipeline fonksiyonu
# ---------------------------------------------------------------------------

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tüm önişleme adımlarını sırayla uygular.
    Döndürülen DataFrame'de şu sütunlar garantidir:
      age_years, gender, height, weight, bmi, ap_hi, ap_lo,
      cholesterol, gluc, smoke, alco, active, cardio
    """
    df = standardize_columns(df)

    # record_id / id sütununu düşür
    for drop_col in ["record_id", "id"]:
        if drop_col in df.columns:
            df = df.drop(columns=[drop_col])

    df = convert_age(df)
    df = calculate_bmi(df)
    df = remove_outliers(df)
    df = handle_missing_values(df)
    df = remove_duplicates(df)

    # Gerekli sütunları seç
    required_cols = [
        "age_years", "gender", "height", "weight", "bmi",
        "ap_hi", "ap_lo", "cholesterol", "gluc",
        "smoke", "alco", "active", "cardio",
    ]

    # Mevcut olanlara göre filtrele
    available = [c for c in required_cols if c in df.columns]
    df = df[available].copy()

    # Tip düzeltmesi
    int_cols = ["gender", "cholesterol", "gluc", "smoke", "alco", "active", "cardio"]
    for col in int_cols:
        if col in df.columns:
            df[col] = df[col].astype(int)

    print(f"[Preprocessing] İşlem tamamlandı. Satır: {len(df)}, Sütun: {df.shape[1]}")
    return df.reset_index(drop=True)


def load_and_preprocess(filepath: str) -> pd.DataFrame:
    """CSV yükleme + ön işleme tek adımda."""
    raw = load_csv(filepath)
    print(f"[Preprocessing] Ham veri yüklendi: {raw.shape}")
    return preprocess(raw)
