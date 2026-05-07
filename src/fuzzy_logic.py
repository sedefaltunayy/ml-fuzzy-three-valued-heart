"""
fuzzy_logic.py
--------------
Bulanık mantık dönüşüm modülü.

calculate_other_factors:
  smoke, alco, active → other_factors (0.0, 0.5, 1.0)

  Tablo 8 esaslı kural:
    • Üç değer de aynıysa:
        smoke=0, alco=0, active=0 → other_factors = 0  (aktif değil, sigara/alkol yok → düşük risk faktörü)
        smoke=1, alco=1, active=1 → other_factors = 1  (tüm olumsuz faktörler var)
        smoke=0, alco=0, active=1 → other_factors = 0  (sağlıklı yaşam tarzı)
    • Değerler karışıksa → other_factors = 0.5 (belirsiz durum)

  Özet:
    - Eğer smoke=0 AND alco=0 AND active=1  → 0.0  (en sağlıklı)
    - Eğer smoke=1 AND alco=1 AND active=0  → 1.0  (en riskli)
    - Eğer smoke=0 AND alco=0 AND active=0  → 0.5  (pasif ama temiz)
    - Diğer tüm kombinasyonlar              → 0.5  (belirsiz)
"""

import pandas as pd
import numpy as np


def calculate_other_factors(row) -> float:
    """
    Tek bir satır (row) için other_factors değerini hesaplar.

    Parametreler
    ------------
    row : pd.Series veya dict-like
        'smoke', 'alco', 'active' anahtarlarını içermeli.

    Döndürür
    --------
    float : 0.0, 0.5 veya 1.0
    """
    smoke  = int(row.get("smoke",  0) if isinstance(row, dict) else row["smoke"])
    alco   = int(row.get("alco",   0) if isinstance(row, dict) else row["alco"])
    active = int(row.get("active", 0) if isinstance(row, dict) else row["active"])

    # En sağlıklı kombinasyon: sigara yok, alkol yok, fiziksel olarak aktif
    if smoke == 0 and alco == 0 and active == 1:
        return 0.0

    # En riskli kombinasyon: sigara var, alkol var, fiziksel olarak inaktif
    if smoke == 1 and alco == 1 and active == 0:
        return 1.0

    # Diğer tüm durumlar – belirsiz (bulanık orta değer)
    return 0.5


def apply_fuzzy_other_factors(df: pd.DataFrame) -> pd.DataFrame:
    """
    DataFrame'deki her satır için other_factors hesaplar ve sütun olarak ekler.
    'smoke', 'alco', 'active' sütunlarının mevcut olmasını bekler.
    """
    df = df.copy()
    df["other_factors"] = df.apply(calculate_other_factors, axis=1)
    return df


def get_other_factors_from_values(smoke: int, alco: int, active: int) -> float:
    """
    Üç değişkeni doğrudan alarak other_factors döndürür.
    Streamlit UI'dan çağrılmak üzere kullanışlı yardımcı.
    """
    return calculate_other_factors({"smoke": smoke, "alco": alco, "active": active})


def describe_other_factors(value: float) -> str:
    """
    other_factors değerine göre açıklama döndürür.
    """
    if value == 0.0:
        return "Sağlıklı yaşam tarzı (sigara/alkol yok, aktif)"
    elif value == 0.5:
        return "Karma/belirsiz yaşam tarzı (bulanık durum)"
    else:
        return "Riskli yaşam tarzı (sigara ve/veya alkol, inaktif)"
