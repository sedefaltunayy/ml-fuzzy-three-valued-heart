"""
visualization.py
----------------
Matplotlib/Seaborn tabanlı grafik üretim fonksiyonları.

Üretilen grafikler:
  outputs/figures/accuracy_comparison.png
  outputs/figures/precision_comparison.png
  outputs/figures/computation_time_comparison.png
  outputs/figures/accuracy_gain.png
  outputs/figures/class_distribution_original.png
  outputs/figures/class_distribution_fuzzy.png
  outputs/figures/feature_correlation.png
"""

import os
import warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Tema
# ---------------------------------------------------------------------------
PALETTE_ORIGINAL = "#4A90D9"
PALETTE_FUZZY    = "#E67E22"
PALETTE_GAIN     = "#2ecc71"
FIG_DPI          = 130
FONT_TITLE       = {"fontsize": 14, "fontweight": "bold"}
FONT_LABEL       = {"fontsize": 11}


def _ensure_output_dir() -> None:
    os.makedirs("outputs/figures", exist_ok=True)


# ---------------------------------------------------------------------------
# Yardımcı: pivot metrikleri
# ---------------------------------------------------------------------------

def _pivot_metric(metrics_df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """
    model_comparison.csv formatındaki DataFrame'i pivot eder.
    Satırlar: model_name, Sütunlar: original / fuzzy
    """
    pivot = metrics_df.pivot_table(
        index="model_name", columns="dataset_type", values=metric
    ).reset_index()
    # Sütunları garanti altına al
    for col in ["original", "fuzzy"]:
        if col not in pivot.columns:
            pivot[col] = 0.0
    return pivot


# ---------------------------------------------------------------------------
# 1. Accuracy Karşılaştırma
# ---------------------------------------------------------------------------

def plot_accuracy_comparison(metrics_df: pd.DataFrame,
                              save: bool = True) -> str:
    """Grouped bar chart: her model için original vs fuzzy accuracy."""
    _ensure_output_dir()
    pivot = _pivot_metric(metrics_df, "accuracy")

    x = np.arange(len(pivot))
    width = 0.35

    fig, ax = plt.subplots(figsize=(11, 6))
    bars1 = ax.bar(x - width/2, pivot["original"], width,
                   color=PALETTE_ORIGINAL, label="Original (Binary)", alpha=0.88)
    bars2 = ax.bar(x + width/2, pivot["fuzzy"],    width,
                   color=PALETTE_FUZZY,    label="Fuzzy (3-Class)", alpha=0.88)

    ax.set_xticks(x)
    ax.set_xticklabels(pivot["model_name"], rotation=25, ha="right")
    ax.set_ylabel("Accuracy", **FONT_LABEL)
    ax.set_title("Model Accuracy: Original vs Fuzzy Dataset", **FONT_TITLE)
    ax.set_ylim(0, 1.1)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.01,
                f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=8)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.01,
                f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    path = "outputs/figures/accuracy_comparison.png"
    if save:
        plt.savefig(path, dpi=FIG_DPI)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# 2. Precision Karşılaştırma
# ---------------------------------------------------------------------------

def plot_precision_comparison(metrics_df: pd.DataFrame,
                               save: bool = True) -> str:
    _ensure_output_dir()
    pivot = _pivot_metric(metrics_df, "precision_score")

    x     = np.arange(len(pivot))
    width = 0.35

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(x - width/2, pivot["original"], width,
           color=PALETTE_ORIGINAL, label="Original (Binary)", alpha=0.88)
    ax.bar(x + width/2, pivot["fuzzy"],    width,
           color=PALETTE_FUZZY,    label="Fuzzy (3-Class)", alpha=0.88)

    ax.set_xticks(x)
    ax.set_xticklabels(pivot["model_name"], rotation=25, ha="right")
    ax.set_ylabel("Precision", **FONT_LABEL)
    ax.set_title("Model Precision: Original vs Fuzzy Dataset", **FONT_TITLE)
    ax.set_ylim(0, 1.1)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()

    path = "outputs/figures/precision_comparison.png"
    if save:
        plt.savefig(path, dpi=FIG_DPI)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# 3. Hesaplama Süresi Karşılaştırma
# ---------------------------------------------------------------------------

def plot_computation_time(metrics_df: pd.DataFrame,
                           save: bool = True) -> str:
    _ensure_output_dir()
    pivot = _pivot_metric(metrics_df, "computation_time")

    x     = np.arange(len(pivot))
    width = 0.35

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(x - width/2, pivot["original"], width,
           color=PALETTE_ORIGINAL, label="Original (Binary)", alpha=0.88)
    ax.bar(x + width/2, pivot["fuzzy"],    width,
           color=PALETTE_FUZZY,    label="Fuzzy (3-Class)", alpha=0.88)

    ax.set_xticks(x)
    ax.set_xticklabels(pivot["model_name"], rotation=25, ha="right")
    ax.set_ylabel("Süre (saniye)", **FONT_LABEL)
    ax.set_title("Model Computation Time: Original vs Fuzzy Dataset", **FONT_TITLE)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()

    path = "outputs/figures/computation_time_comparison.png"
    if save:
        plt.savefig(path, dpi=FIG_DPI)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# 4. Accuracy Gain (Fuzzy – Original)
# ---------------------------------------------------------------------------

def plot_accuracy_gain(metrics_df: pd.DataFrame,
                       save: bool = True) -> str:
    """
    Her model için (Fuzzy Accuracy – Original Accuracy) bar grafiği.
    Pozitif değer: fuzzy daha iyi; negatif: original daha iyi.
    """
    _ensure_output_dir()
    pivot = _pivot_metric(metrics_df, "accuracy")
    pivot["gain"] = pivot["fuzzy"] - pivot["original"]

    colors = [PALETTE_GAIN if g >= 0 else "#e74c3c" for g in pivot["gain"]]

    fig, ax = plt.subplots(figsize=(11, 6))
    bars = ax.bar(pivot["model_name"], pivot["gain"], color=colors, alpha=0.88)

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Model", **FONT_LABEL)
    ax.set_ylabel("Accuracy Gain (Fuzzy – Original)", **FONT_LABEL)
    ax.set_title("Accuracy Gain: Fuzzy vs Original Dataset", **FONT_TITLE)
    ax.set_xticklabels(pivot["model_name"], rotation=25, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    for bar, val in zip(bars, pivot["gain"]):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + (0.002 if val >= 0 else -0.01),
                f"{val:+.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    # Akademik not
    ax.text(0.01, 0.98,
            "Not: Fuzzy modelde yüksek accuracy, kural tabanlı hedefi\n"
            "öğrenme başarısının göstergesidir (beklenen davranış).",
            transform=ax.transAxes, fontsize=8, color="gray",
            verticalalignment="top", style="italic")

    plt.tight_layout()
    path = "outputs/figures/accuracy_gain.png"
    if save:
        plt.savefig(path, dpi=FIG_DPI)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# 5. Sınıf Dağılımı – Original
# ---------------------------------------------------------------------------

def plot_class_distribution_original(df: pd.DataFrame,
                                      save: bool = True) -> str:
    _ensure_output_dir()
    if "cardio" not in df.columns:
        return ""

    counts = df["cardio"].value_counts().sort_index()
    labels = {0: "Hastalık Yok (0)", 1: "Hastalık Var (1)"}
    clabels = [labels.get(i, str(i)) for i in counts.index]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(clabels, counts.values, color=[PALETTE_ORIGINAL, "#e74c3c"], alpha=0.88)
    ax.set_title("Original Dataset – Hedef Dağılımı (Binary)", **FONT_TITLE)
    ax.set_ylabel("Örnek Sayısı", **FONT_LABEL)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    for i, v in enumerate(counts.values):
        ax.text(i, v + 100, str(v), ha="center", fontsize=10)
    plt.tight_layout()

    path = "outputs/figures/class_distribution_original.png"
    if save:
        plt.savefig(path, dpi=FIG_DPI)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# 6. Sınıf Dağılımı – Fuzzy
# ---------------------------------------------------------------------------

def plot_class_distribution_fuzzy(df: pd.DataFrame,
                                   save: bool = True) -> str:
    _ensure_output_dir()
    if "fuzzy_target" not in df.columns:
        return ""

    counts = df["fuzzy_target"].value_counts().sort_index()
    label_map = {0: "Risk Yok (0)", 1: "Risk Var (1)", 2: "Risk Olabilir (2)"}
    clabels = [label_map.get(i, str(i)) for i in counts.index]
    colors  = ["#2ecc71", "#e74c3c", "#f39c12"]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(clabels, counts.values,
           color=colors[:len(counts)], alpha=0.88)
    ax.set_title("Fuzzy Dataset – Hedef Dağılımı (3-Değerli)", **FONT_TITLE)
    ax.set_ylabel("Örnek Sayısı", **FONT_LABEL)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    for i, v in enumerate(counts.values):
        ax.text(i, v + 100, str(v), ha="center", fontsize=10)
    plt.tight_layout()

    path = "outputs/figures/class_distribution_fuzzy.png"
    if save:
        plt.savefig(path, dpi=FIG_DPI)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# 7. Feature Korelasyon Isı Haritası
# ---------------------------------------------------------------------------

def plot_feature_correlation(df: pd.DataFrame,
                              save: bool = True) -> str:
    _ensure_output_dir()
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.shape[1] < 2:
        return ""

    corr = numeric_df.corr()
    fig, ax = plt.subplots(figsize=(12, 9))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
                center=0, linewidths=0.5, ax=ax,
                annot_kws={"size": 8})
    ax.set_title("Özellik Korelasyon Isı Haritası", **FONT_TITLE)
    plt.tight_layout()

    path = "outputs/figures/feature_correlation.png"
    if save:
        plt.savefig(path, dpi=FIG_DPI)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Tüm grafikleri üret
# ---------------------------------------------------------------------------

def generate_all_figures(metrics_df: pd.DataFrame,
                          df_original: pd.DataFrame = None,
                          df_fuzzy: pd.DataFrame = None) -> list[str]:
    """
    Tüm standart grafikleri üretir ve yollarını döndürür.
    """
    paths = []
    paths.append(plot_accuracy_comparison(metrics_df))
    paths.append(plot_precision_comparison(metrics_df))
    paths.append(plot_computation_time(metrics_df))
    paths.append(plot_accuracy_gain(metrics_df))

    if df_original is not None:
        paths.append(plot_class_distribution_original(df_original))
        paths.append(plot_feature_correlation(df_original))

    if df_fuzzy is not None:
        paths.append(plot_class_distribution_fuzzy(df_fuzzy))

    paths = [p for p in paths if p]
    print(f"[Visualization] {len(paths)} grafik üretildi.")
    return paths
