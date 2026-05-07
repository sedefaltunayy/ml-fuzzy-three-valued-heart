"""
train.py
--------
7 ML modelini iki veri seti (Original binary + Fuzzy three-valued) üzerinde
sklearn Pipeline ile eğiten, değerlendiren ve kaydeden modül.

Algoritmalar (makale parametreleri):
  - GaussianNB
  - SVC  (Linear kernel, StandardScaler gerektirir)
  - AdaBoost (n_estimators=100)
  - DecisionTree (max_depth=10)
  - KNN (n_neighbors=15, StandardScaler gerektirir)
  - RandomForest (n_estimators=25, max_features=min(4, n_features))
  - GradientBoosting (n_estimators=90)

Çıktılar:
  models/original/{model_name}.joblib
  models/fuzzy/{model_name}.joblib
  outputs/metrics/model_comparison.csv
  outputs/reports/classification_report_original_best.csv
  outputs/reports/classification_report_fuzzy_best.csv
  outputs/figures/confusion_matrix_original_best.png
  outputs/figures/confusion_matrix_fuzzy_best.png
"""

import os
import time
import warnings
import joblib
import pandas as pd
import numpy as np

from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Değerlendirme fonksiyonları evaluate.py'den içe aktarılır
from .evaluate import (
    evaluate_model,
    save_classification_report,
    save_confusion_matrix_plot,
    print_evaluation_summary,
)

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Klasör yapısı
# ---------------------------------------------------------------------------

DIRS = [
    "data", "database",
    "models/original", "models/fuzzy",
    "outputs/figures", "outputs/reports", "outputs/metrics",
]


def ensure_dirs() -> None:
    """Gerekli tüm klasörleri oluşturur."""
    for d in DIRS:
        os.makedirs(d, exist_ok=True)


# ---------------------------------------------------------------------------
# Model tanımları
# ---------------------------------------------------------------------------

def build_model_definitions(n_features: int) -> dict:
    """
    n_features: X_train'deki özellik sayısı (RF max_features kısıtı için)
    """
    safe_max_features = min(4, n_features)

    return {
        "GaussianNB": GaussianNB(),

        "SVM_Linear": Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    SVC(kernel="linear", probability=True, random_state=42)),
        ]),

        "AdaBoost": AdaBoostClassifier(
            n_estimators=100, random_state=42, algorithm="SAMME"
        ),

        "DecisionTree": DecisionTreeClassifier(
            max_depth=10, random_state=42
        ),

        "KNN": Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    KNeighborsClassifier(n_neighbors=15)),
        ]),

        "RandomForest": RandomForestClassifier(
            n_estimators=25,
            max_features=safe_max_features,
            random_state=42,
        ),

        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=90, random_state=42
        ),
    }


# ---------------------------------------------------------------------------
# Özellik seçimi
# ---------------------------------------------------------------------------

ORIGINAL_FEATURES = [
    "age_years", "gender", "height", "weight", "bmi",
    "ap_hi", "ap_lo", "cholesterol", "gluc",
    "smoke", "alco", "active",
]

FUZZY_FEATURES = [
    "age_years", "gender", "height", "weight", "bmi",
    "ap_hi", "ap_lo", "cholesterol", "gluc",
    "other_factors",
]


def get_features_and_target(df: pd.DataFrame, dataset_type: str):
    """
    dataset_type: 'original' veya 'fuzzy'
    Döndürür: X (DataFrame), y (Series)
    """
    if dataset_type == "original":
        feature_cols = [c for c in ORIGINAL_FEATURES if c in df.columns]
        target_col   = "cardio"
    else:
        feature_cols = [c for c in FUZZY_FEATURES if c in df.columns]
        target_col   = "fuzzy_target"

    if target_col not in df.columns:
        raise KeyError(f"Hedef sütun '{target_col}' DataFrame'de bulunamadı.")

    X = df[feature_cols]
    y = df[target_col]
    return X, y


# ---------------------------------------------------------------------------
# Train/test split (stratify ile fallback)
# ---------------------------------------------------------------------------

def safe_train_test_split(X, y, test_size: float = 0.2, random_state: int = 42):
    """
    stratify=y ile split dener; sınıf başına örnek yetersizse stratify=None ile tekrar dener.
    """
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        return X_train, X_test, y_train, y_test
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
# Metrik hesaplama
# ---------------------------------------------------------------------------

def compute_metrics(y_true, y_pred, model_name: str,
                    dataset_type: str, elapsed: float) -> dict:
    """
    Tüm temel metrikleri hesaplar.
    evaluate.evaluate_model'e proxy – geriye dönük uyumluluk için saklandı.
    """
    return evaluate_model(y_true, y_pred, model_name, dataset_type, elapsed)


# ---------------------------------------------------------------------------
# Tek model eğitimi
# ---------------------------------------------------------------------------

def train_single_model(model, model_name: str,
                       X_train, X_test, y_train, y_test,
                       dataset_type: str, save_dir: str) -> dict:
    """
    Modeli eğitir, değerlendirir, joblib ile kaydeder.
    Döndürür: metrik sözlüğü + y_pred
    """
    start = time.time()
    model.fit(X_train, y_train)
    elapsed = time.time() - start

    y_pred = model.predict(X_test)
    metrics = compute_metrics(y_test, y_pred, model_name, dataset_type, elapsed)

    # Joblib kayıt
    ensure_dirs()
    model_path = os.path.join(save_dir, f"{model_name}.joblib")
    joblib.dump(model, model_path)
    print(f"  [Model] {model_path} kaydedildi. Accuracy={metrics['accuracy']:.4f}")

    return metrics, y_pred


# ---------------------------------------------------------------------------
# Tüm modelleri eğit – tek dataset
# ---------------------------------------------------------------------------

def train_all_models(df: pd.DataFrame, dataset_type: str) -> list[dict]:
    """
    Belirtilen dataset_type ('original' | 'fuzzy') üzerinde 7 modeli eğitir.

    Döndürür: her modele ait metrik sözlüklerinin listesi
    """
    ensure_dirs()
    X, y = get_features_and_target(df, dataset_type)
    X_train, X_test, y_train, y_test = safe_train_test_split(X, y)

    n_features = X_train.shape[1]
    models      = build_model_definitions(n_features)
    save_dir    = f"models/{dataset_type}"
    all_metrics = []

    best_acc      = -1
    best_name     = None
    best_y_pred   = None

    print(f"\n{'='*60}")
    print(f"[Train] Dataset: {dataset_type.upper()} | "
          f"Özellik: {n_features} | Satır: {len(X_train)} (train) / {len(X_test)} (test)")
    print(f"{'='*60}")

    for name, model in models.items():
        print(f"\n  >> {name}")
        metrics, y_pred = train_single_model(
            model, name, X_train, X_test, y_train, y_test,
            dataset_type, save_dir
        )
        all_metrics.append(metrics)

        if metrics["accuracy"] > best_acc:
            best_acc    = metrics["accuracy"]
            best_name   = name
            best_y_pred = y_pred

    # En iyi model için zorunlu çıktılar
    _save_best_model_outputs(
        y_test, best_y_pred, best_name, dataset_type
    )

    return all_metrics


# ---------------------------------------------------------------------------
# En iyi model çıktıları
# ---------------------------------------------------------------------------

def _save_best_model_outputs(y_test, y_pred, model_name: str,
                              dataset_type: str) -> None:
    """Classification report ve confusion matrix'i dosyaya kaydeder."""
    ensure_dirs()

    report_path = (
        f"outputs/reports/classification_report_{dataset_type}_best.csv"
    )
    cm_path = (
        f"outputs/figures/confusion_matrix_{dataset_type}_best.png"
    )

    # Fuzzy modelde daha açıklayıcı etiketler
    label_map = None
    if dataset_type == "fuzzy":
        label_map = {0: "Risk Yok", 1: "Risk Var", 2: "Risk Olabilir"}

    save_classification_report(y_test, y_pred, report_path, label_map=label_map)
    save_confusion_matrix_plot(y_test, y_pred, model_name, dataset_type, cm_path, label_map=label_map)

    print(f"\n  [Best Model] {model_name} – Accuracy={accuracy_score(y_test, y_pred):.4f}")


# ---------------------------------------------------------------------------
# Her iki dataset için eğitim + karşılaştırma CSV
# ---------------------------------------------------------------------------

def run_full_training(df_original: pd.DataFrame,
                      df_fuzzy: pd.DataFrame) -> pd.DataFrame:
    """
    Original ve Fuzzy dataset üzerinde tüm modelleri eğitir.
    Karşılaştırma metriklerini CSV olarak kaydeder.

    Döndürür: birleşik metrik DataFrame
    """
    ensure_dirs()

    print("\n[PHASE 1] Original Dataset Eğitimi...")
    metrics_original = train_all_models(df_original, "original")

    print("\n[PHASE 2] Fuzzy Dataset Eğitimi...")
    metrics_fuzzy = train_all_models(df_fuzzy, "fuzzy")

    all_metrics = metrics_original + metrics_fuzzy
    metrics_df  = pd.DataFrame(all_metrics)

    # Karşılaştırma CSV
    metrics_path = "outputs/metrics/model_comparison.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"\n[Metrics] {metrics_path} kaydedildi.")

    return metrics_df


# ---------------------------------------------------------------------------
# Eğitilmiş modeli yükle
# ---------------------------------------------------------------------------

def load_model(model_name: str, dataset_type: str):
    """Kayıtlı joblib modelini yükler."""
    path = os.path.join(f"models/{dataset_type}", f"{model_name}.joblib")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model bulunamadı: {path}")
    return joblib.load(path)


def list_trained_models(dataset_type: str) -> list[str]:
    """Kaydedilmiş modellerin isimlerini döndürür."""
    dir_path = f"models/{dataset_type}"
    if not os.path.exists(dir_path):
        return []
    return [
        f.replace(".joblib", "")
        for f in os.listdir(dir_path)
        if f.endswith(".joblib")
    ]
