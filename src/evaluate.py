"""
evaluate.py
-----------
Modüler değerlendirme fonksiyonları.

- evaluate_model       : sklearn tahmin sonuçlarından tüm metrikleri hesaplar
- save_classification_report  : CSV'ye classification report yazar
- save_confusion_matrix_plot  : PNG'ye confusion matrix yazar
- print_evaluation_summary    : Konsola özet tablo basar
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix,
)


# ---------------------------------------------------------------------------
# Temel metrik hesaplama
# ---------------------------------------------------------------------------

def evaluate_model(
    y_true,
    y_pred,
    model_name: str = "Model",
    dataset_type: str = "original",
    computation_time: float = 0.0,
) -> dict:
    """
    Tahmin sonuçlarından temel metrikleri hesaplar ve sözlük olarak döndürür.

    Parametreler
    ------------
    y_true           : gerçek etiketler
    y_pred           : model tahminleri
    model_name       : model adı (kayıt için)
    dataset_type     : 'original' veya 'fuzzy'
    computation_time : eğitim süresi (saniye)

    Döndürür
    --------
    dict: dataset_type, model_name, accuracy, f1_score, precision_score,
          recall_score, computation_time
    """
    unique_classes = set(y_true)
    avg_method = "binary" if len(unique_classes) <= 2 else "weighted"

    return {
        "dataset_type":      dataset_type,
        "model_name":        model_name,
        "accuracy":          round(float(accuracy_score(y_true, y_pred)), 4),
        "f1_score":          round(float(f1_score(
                                 y_true, y_pred,
                                 average=avg_method, zero_division=0
                             )), 4),
        "precision_score":   round(float(precision_score(
                                 y_true, y_pred,
                                 average=avg_method, zero_division=0
                             )), 4),
        "recall_score":      round(float(recall_score(
                                 y_true, y_pred,
                                 average=avg_method, zero_division=0
                             )), 4),
        "computation_time":  round(computation_time, 4),
    }


# ---------------------------------------------------------------------------
# Classification report → CSV
# ---------------------------------------------------------------------------

def save_classification_report(
    y_true,
    y_pred,
    filepath: str,
    label_map: dict = None,
) -> None:
    """
    sklearn classification_report çıktısını CSV olarak kaydeder.

    Parametreler
    ------------
    y_true     : gerçek etiketler
    y_pred     : model tahminleri
    filepath   : kayıt yolu  (örn: outputs/reports/classification_report_original_best.csv)
    label_map  : {0: "Risk Yok", 1: "Risk Var", 2: "Risk Olabilir"} gibi etiket haritası
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    target_names = None
    if label_map:
        labels_sorted = sorted(label_map.keys())
        target_names  = [str(label_map[l]) for l in labels_sorted]

    report_dict = classification_report(
        y_true, y_pred,
        output_dict=True,
        zero_division=0,
        target_names=target_names,
    )
    report_df = pd.DataFrame(report_dict).transpose()
    report_df.to_csv(filepath)
    print(f"  [Report] Kaydedildi: {filepath}")


# ---------------------------------------------------------------------------
# Confusion matrix → PNG
# ---------------------------------------------------------------------------

def save_confusion_matrix_plot(
    y_true,
    y_pred,
    model_name: str,
    dataset_type: str,
    filepath: str,
    label_map: dict = None,
) -> None:
    """
    Confusion matrix'i seaborn heatmap olarak PNG kaydeder.

    Parametreler
    ------------
    y_true       : gerçek etiketler
    y_pred       : model tahminleri
    model_name   : başlıkta kullanılacak model adı
    dataset_type : 'original' veya 'fuzzy'
    filepath     : PNG kayıt yolu
    label_map    : eksen etiketleri için  {0: "Risk Yok", ...}
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    labels = sorted(set(y_true) | set(y_pred))
    cm     = confusion_matrix(y_true, y_pred, labels=labels)

    tick_labels = labels
    if label_map:
        tick_labels = [str(label_map.get(l, l)) for l in labels]

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=tick_labels,
        yticklabels=tick_labels,
        ax=ax,
    )
    ax.set_xlabel("Tahmin Edilen", fontsize=11)
    ax.set_ylabel("Gerçek", fontsize=11)
    ax.set_title(
        f"Confusion Matrix – {model_name}\n({dataset_type.capitalize()} Dataset)",
        fontsize=12,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(filepath, dpi=130)
    plt.close(fig)
    print(f"  [Figure] Kaydedildi: {filepath}")


# ---------------------------------------------------------------------------
# Konsol özet tablosu
# ---------------------------------------------------------------------------

def print_evaluation_summary(metrics_list: list[dict]) -> None:
    """
    Metrik sözlükleri listesini konsola tablo olarak basar.

    Parametreler
    ------------
    metrics_list : [{"dataset_type":..., "model_name":..., "accuracy":..., ...}, ...]
    """
    if not metrics_list:
        print("[Evaluate] Metrik listesi boş.")
        return

    df = pd.DataFrame(metrics_list)
    display_cols = [
        "dataset_type", "model_name",
        "accuracy", "f1_score", "precision_score",
        "recall_score", "computation_time",
    ]
    available = [c for c in display_cols if c in df.columns]
    df_sorted = df[available].sort_values(
        ["dataset_type", "accuracy"], ascending=[True, False]
    )

    print("\n" + "=" * 80)
    print("MODEL DEĞERLENDİRME ÖZETİ")
    print("=" * 80)
    print(df_sorted.to_string(index=False))
    print("=" * 80 + "\n")
