"""
run_pipeline.py
---------------
CLI aracılığıyla tüm pipeline'ı tek komutla çalıştıran script.

Kullanım:
    python scripts/run_pipeline.py --csv data/cardio_train.csv

Adımlar:
  1. Klasörleri oluştur
  2. CSV yükle → raw_heart_data tablosuna yaz
  3. Preprocessing → processed_original_data tablosuna yaz
  4. Fuzzy dönüşümü → fuzzy_modified_data tablosuna yaz
  5. 7 modeli eğit (original + fuzzy)
  6. Metrikleri → model_results tablosuna yaz
  7. Tüm grafikleri üret
  8. Özet raporu yaz
"""

import argparse
import os
import sys
import time

# Projenin kök dizinine path ekle
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.database       import DatabaseManager
from src.preprocessing  import load_csv, preprocess
from src.fuzzy_logic    import apply_fuzzy_other_factors
from src.rules          import apply_rules_to_dataframe
from src.train          import run_full_training, ensure_dirs
from src.visualization  import generate_all_figures


# ---------------------------------------------------------------------------
# Klasör yapısı
# ---------------------------------------------------------------------------

REQUIRED_DIRS = [
    "data", "database",
    "models/original", "models/fuzzy",
    "outputs/figures", "outputs/reports", "outputs/metrics",
]


def create_directories() -> None:
    """Gerekli tüm dizinleri oluşturur."""
    for d in REQUIRED_DIRS:
        os.makedirs(d, exist_ok=True)
    print("[Setup] Klasörler oluşturuldu.")


# ---------------------------------------------------------------------------
# Ana pipeline
# ---------------------------------------------------------------------------

def run_pipeline(csv_path: str) -> None:
    total_start = time.time()

    # 0. Klasörler
    create_directories()
    ensure_dirs()

    # 1. Veritabanı
    db = DatabaseManager()
    db.create_tables()
    print("[DB] Tablolar oluşturuldu.")

    # 2. Ham veri yükle
    print(f"\n[Step 1] CSV yükleniyor: {csv_path}")
    raw_df = load_csv(csv_path)
    print(f"  Ham veri: {raw_df.shape}")
    db.to_sql(raw_df, "raw_heart_data", if_exists="replace")
    print("  → raw_heart_data tablosuna yazıldı.")

    # 3. Preprocessing
    print("\n[Step 2] Preprocessing...")
    processed_df = preprocess(raw_df)
    db.to_sql(processed_df, "processed_original_data", if_exists="replace")
    print(f"  → processed_original_data: {processed_df.shape}")

    # 4. Fuzzy dönüşümü
    print("\n[Step 3] Fuzzy dönüşümü...")
    fuzzy_df = apply_fuzzy_other_factors(processed_df)
    fuzzy_df = apply_rules_to_dataframe(fuzzy_df)

    # Fuzzy tablo için gerekli sütunlar
    fuzzy_cols = [
        "age_years", "gender", "height", "weight", "bmi",
        "ap_hi", "ap_lo", "cholesterol", "gluc",
        "other_factors", "fuzzy_target",
    ]
    fuzzy_save = fuzzy_df[[c for c in fuzzy_cols if c in fuzzy_df.columns]]
    db.to_sql(fuzzy_save, "fuzzy_modified_data", if_exists="replace")
    print(f"  → fuzzy_modified_data: {fuzzy_save.shape}")
    print(f"  fuzzy_target dağılımı:\n{fuzzy_df['fuzzy_target'].value_counts().sort_index()}")

    # 5. Model eğitimi
    print("\n[Step 4] Model eğitimi başlıyor...")
    metrics_df = run_full_training(processed_df, fuzzy_df)

    # Model sonuçlarını DB'ye yaz (önceki sonuçları temizle)
    db.clear_table("model_results")
    db.to_sql(metrics_df, "model_results", if_exists="append")
    print(f"  → model_results tablosuna {len(metrics_df)} satır yazıldı.")

    # 6. Grafikler
    print("\n[Step 5] Grafikler üretiliyor...")
    generate_all_figures(metrics_df, processed_df, fuzzy_df)

    # 7. Özet
    total_elapsed = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"[Pipeline TAMAMLANDI] Toplam süre: {total_elapsed:.1f} saniye")
    print(f"{'='*60}")

    print("\n  Çıktı dosyaları:")
    for root, _, files in os.walk("outputs"):
        for f in files:
            print(f"    {os.path.join(root, f)}")
    for root, _, files in os.walk("models"):
        for f in files:
            print(f"    {os.path.join(root, f)}")


# ---------------------------------------------------------------------------
# CLI giriş noktası
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="HibritML – Fuzzy Üç Değerli Mantık Pipeline"
    )
    parser.add_argument(
        "--csv",
        type=str,
        required=True,
        help="Giriş CSV dosyasının yolu (örn: data/cardio_train.csv)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if not os.path.exists(args.csv):
        print(f"[HATA] CSV dosyası bulunamadı: {args.csv}")
        sys.exit(1)

    run_pipeline(args.csv)
