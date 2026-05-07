"""
rules.py
--------
12 Karar kuralına dayalı bulanık hedef değişken (fuzzy_target) üretim motoru.

Değişkenler
-----------
  gender       : 1 = Kadın, 2 = Erkek
  age_years    : yaş (yıl cinsinden)
  other_factors: 0.0 = sağlıklı, 0.5 = belirsiz, 1.0 = riskli

Çıktı (fuzzy_target)
---------------------
  0 – Risk Yok       (yeşil)
  1 – Risk Olabilir  (turuncu)
  2 – Risk Var       (kırmızı)

12 Kural Tablosu
----------------
Kural 1 : Kadın  + Genç (<40)         + OF=0.0 → 0
Kural 2 : Kadın  + Genç (<40)         + OF=0.5 → 1
Kural 3 : Kadın  + Genç (<40)         + OF=1.0 → 1
Kural 4 : Kadın  + Orta yaş (40-60)   + OF=0.0 → 1
Kural 5 : Kadın  + Orta yaş (40-60)   + OF=0.5 → 1
Kural 6 : Kadın  + Orta yaş (40-60)   + OF=1.0 → 2
Kural 7 : Erkek  + Genç (<40)         + OF=0.0 → 0
Kural 8 : Erkek  + Genç (<40)         + OF=0.5 → 1
Kural 9 : Erkek  + Genç (<40)         + OF=1.0 → 2
Kural 10: Erkek  + Orta yaş (40-60)   + OF=0.0 → 1
Kural 11: Erkek  + Orta yaş (40-60)   + OF=0.5 → 2
Kural 12: Yaşlı  (>60)                + herhangi → 2
"""

import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# Sabitler
# ---------------------------------------------------------------------------

FUZZY_LABELS = {
    0: "Risk Yok",
    1: "Risk Olabilir",
    2: "Risk Var",
}

FUZZY_COLORS = {
    0: "#2ecc71",   # yeşil
    1: "#f39c12",   # turuncu
    2: "#e74c3c",   # kırmızı
}


# ---------------------------------------------------------------------------
# Tek satır kural motoru
# ---------------------------------------------------------------------------

def apply_rules(row) -> int:
    """
    Tek bir kayıt için 12 kurala göre fuzzy_target (0, 1, 2) döndürür.

    Parametreler
    ------------
    row : pd.Series veya dict-like
        'gender', 'age_years', 'other_factors' anahtarları gerekli.
    """
    gender = int(row["gender"])
    age    = float(row["age_years"])
    of     = float(row["other_factors"])

    # Kural 12 – Yaşlı (>60): her durumda risk var
    if age > 60:
        return 2

    # Kadın (gender == 1)
    if gender == 1:
        if age < 40:
            if of == 0.0:
                return 0   # Kural 1
            else:
                return 1   # Kural 2 & 3
        else:  # 40 <= age <= 60
            if of == 0.0:
                return 1   # Kural 4
            elif of == 0.5:
                return 1   # Kural 5
            else:          # of == 1.0
                return 2   # Kural 6

    # Erkek (gender == 2)
    if gender == 2:
        if age < 40:
            if of == 0.0:
                return 0   # Kural 7
            elif of == 0.5:
                return 1   # Kural 8
            else:           # of == 1.0
                return 2   # Kural 9
        else:  # 40 <= age <= 60
            if of == 0.0:
                return 1   # Kural 10
            else:           # of == 0.5 veya 1.0
                return 2   # Kural 11

    # Bilinmeyen cinsiyet → risk belirsiz
    return 1


def apply_rules_to_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    DataFrame'deki her satır için apply_rules çağırır ve
    'fuzzy_target' sütununu ekler.
    """
    df = df.copy()
    df["fuzzy_target"] = df.apply(apply_rules, axis=1)
    return df


def get_fuzzy_label(target_value: int) -> str:
    """fuzzy_target int değerini insan okunabilir etikete çevirir."""
    return FUZZY_LABELS.get(int(target_value), "Bilinmiyor")


def get_fuzzy_color(target_value: int) -> str:
    """fuzzy_target için renk kodu döndürür (Streamlit/HTML)."""
    return FUZZY_COLORS.get(int(target_value), "#95a5a6")


def predict_single(gender: int, age_years: float,
                   other_factors: float) -> dict:
    """
    Tek bir kullanıcı için kural tabanlı fuzzy karar verir.

    Döndürür
    --------
    dict:
        fuzzy_target (int), label (str), color (str), rules_applied (str)
    """
    row = {
        "gender":        gender,
        "age_years":     age_years,
        "other_factors": other_factors,
    }
    target = apply_rules(row)
    return {
        "fuzzy_target":    target,
        "label":           get_fuzzy_label(target),
        "color":           get_fuzzy_color(target),
        "rules_applied":   _describe_rule(gender, age_years, other_factors, target),
    }


def _describe_rule(gender: int, age_years: float,
                   other_factors: float, result: int) -> str:
    """Hangi kuralın tetiklendiğini açıklar."""
    g_str  = "Kadın" if gender == 1 else "Erkek"
    of_str = {0.0: "Sağlıklı (0.0)", 0.5: "Belirsiz (0.5)", 1.0: "Riskli (1.0)"}.get(
        other_factors, f"{other_factors}"
    )

    if age_years > 60:
        age_group = "Yaşlı (>60)"
        rule_id   = "Kural 12"
    elif age_years < 40:
        age_group = "Genç (<40)"
        rule_id   = "Kural 1-3 / 7-9"
    else:
        age_group = "Orta Yaş (40-60)"
        rule_id   = "Kural 4-6 / 10-11"

    result_str = FUZZY_LABELS[result]
    return (
        f"{rule_id} | {g_str} | {age_group} | "
        f"OtherFactors={of_str} → **{result_str}**"
    )
