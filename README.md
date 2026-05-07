# HibritML – Bulanık Üç Değerli Mantık ile Hibrit ML Modeli

## 📋 Proje Özeti

Klasik binary (0-1) sınıflandırmanın belirsiz durumları (örn: hafif riskli hastalar) temsil edememesi sorununa yönelik akademik çözüm. Kaggle Cardiovascular Disease Dataset üzerinde **bulanık üç değerli mantık** ve **7 ML algoritması** hibridlenerek karşılaştırmalı analiz yapılır.

| Katman | Açıklama |
|---|---|
| **Girdi** | Kaggle Cardiovascular Disease Dataset (`;` veya `,` otomatik algılanır) |
| **Preprocessing** | Yaş dönüşümü (gün→yıl), BMI hesabı, tıbbi outlier temizliği |
| **Fuzzy Girdi** | `other_factors` = f(smoke, alco, active) ∈ {0.0, 0.5, 1.0} |
| **Fuzzy Hedef** | 12 kural: gender + age_years + other_factors → {0, 1, 2} |
| **ML Katmanı** | 7 algoritma × 2 dataset = 14 model eğitimi |
| **Çıktı** | Accuracy, F1, Precision, süre + görselleştirme + SQLite |

---

## 📁 Dosya Yapısı

```
hibritML/
├── app.py                        # Streamlit dashboard (ana giriş)
├── requirements.txt              # Tüm bağımlılıklar
├── README.md
├── src/
│   ├── __init__.py               # Paket başlatıcı (tüm public API)
│   ├── database.py               # SQLite bağlantı sınıfı (5 tablo)
│   ├── preprocessing.py          # Temizlik, BMI, outlier, delimiter detection
│   ├── fuzzy_logic.py            # other_factors hesabı (Tablo 8 esaslı)
│   ├── rules.py                  # 12 karar kuralı motoru
│   ├── train.py                  # 7 model eğitimi (sklearn Pipeline)
│   ├── evaluate.py               # Metrik hesaplama, classification report, CM
│   └── visualization.py          # Matplotlib/Seaborn grafik fonksiyonları
└── scripts/
    └── run_pipeline.py           # CLI tek-komut pipeline
```

Çalışma zamanında **otomatik oluşturulan** dizinler:
```
data/          models/original/    outputs/figures/
database/      models/fuzzy/       outputs/reports/
                                   outputs/metrics/
```

---

## 🛠️ Kurulum

```bash
cd hibritML
pip install -r requirements.txt
```

> Python 3.10+ önerilir.

---

## 🚀 Çalıştırma

### Yöntem A – Streamlit Dashboard (Önerilen)

```bash
streamlit run app.py
```

Dashboard üzerinden sırayla:
1. **📂 Veri Yükle & Pipeline** sayfasına git
2. CSV dosyasını yükle (Kaggle `cardio_train.csv`)
3. **🗄️ SQLite Veritabanı Oluştur** butonuna tıkla
4. **⚙️ Preprocessing Çalıştır** butonuna tıkla
5. **🌀 Fuzzy Dataset Oluştur** butonuna tıkla
6. **🤖 Tüm Modelleri Eğit** butonuna tıkla (birkaç dk)
7. **📊 Model Sonuçları** sayfasında grafikleri incele
8. **🔬 Risk Tahmini** sayfasında hasta bilgisi gir

### Yöntem B – CLI Pipeline (tek komut)

```bash
python scripts/run_pipeline.py --csv data/cardio_train.csv
```

---

## 🗄️ Veritabanı Şeması (SQLite)

**Dosya:** `database/heart_disease.db`

| Tablo | Açıklama | Yazma Stratejisi |
|---|---|---|
| `raw_heart_data` | Yüklenen ham CSV verisi | `replace` |
| `processed_original_data` | Önişlenmiş binary hedef (cardio) | `replace` |
| `fuzzy_modified_data` | other_factors + fuzzy_target sütunları | `replace` |
| `model_results` | 14 modelin accuracy/F1/precision/süre metrikleri | `clear + append` |
| `user_predictions` | Dashboard üzerinden yapılan tahmin geçmişi | `append` |

---

## 🧠 Fuzzy Hedef Encoding

| Kavramsal Değer | Anlam | Model Sınıfı (integer) |
|---|---|---|
| Risk Yok | Sağlıklı durum | **0** |
| Risk Var | Hastalık riski | **1** |
| Risk Olabilir | Belirsiz durum | **2** |

> ⚠️ `0.5 → 2` dönüşümü sadece UI gösterimi içindir. Model eğitiminde integer label (0,1,2) kullanılır.

---

## 🧠 12 Karar Kuralı

| Kural | Cinsiyet | Yaş | OtherFactors | Sonuç |
|---|---|---|---|---|
| 1 | Kadın | < 40 | 0.0 | 0 – Risk Yok |
| 2 | Kadın | < 40 | 0.5 | 1 – Risk Olabilir |
| 3 | Kadın | < 40 | 1.0 | 1 – Risk Olabilir |
| 4 | Kadın | 40–60 | 0.0 | 1 – Risk Olabilir |
| 5 | Kadın | 40–60 | 0.5 | 1 – Risk Olabilir |
| 6 | Kadın | 40–60 | 1.0 | 2 – Risk Var |
| 7 | Erkek | < 40 | 0.0 | 0 – Risk Yok |
| 8 | Erkek | < 40 | 0.5 | 1 – Risk Olabilir |
| 9 | Erkek | < 40 | 1.0 | 2 – Risk Var |
| 10 | Erkek | 40–60 | 0.0 | 1 – Risk Olabilir |
| 11 | Erkek | 40–60 | 0.5/1.0 | 2 – Risk Var |
| 12 | Her ikisi | > 60 | Herhangi | 2 – Risk Var |

---

## 📊 Model Parametreleri

| Model | Parametre |
|---|---|
| GaussianNB | Varsayılan |
| SVM (Linear) | `StandardScaler` + `kernel="linear"` |
| AdaBoost | `n_estimators=100, algorithm="SAMME"` |
| DecisionTree | `max_depth=10` |
| KNN | `StandardScaler` + `n_neighbors=15` |
| RandomForest | `n_estimators=25, max_features=min(4, n_features)` |
| GradientBoosting | `n_estimators=90` |

---

## 📂 Çıktı Dosyaları

```
outputs/metrics/model_comparison.csv
outputs/reports/classification_report_original_best.csv
outputs/reports/classification_report_fuzzy_best.csv
outputs/figures/confusion_matrix_original_best.png
outputs/figures/confusion_matrix_fuzzy_best.png
outputs/figures/accuracy_comparison.png
outputs/figures/precision_comparison.png
outputs/figures/computation_time_comparison.png
outputs/figures/accuracy_gain.png
outputs/figures/class_distribution_original.png
outputs/figures/class_distribution_fuzzy.png
outputs/figures/feature_correlation.png
models/original/{model_name}.joblib
models/fuzzy/{model_name}.joblib
```

---

## ⚠️ Önemli Akademik Not

> **Original model binary target (`cardio` = 0/1) üzerinde eğitilirken, fuzzy-modified model
> rule-derived three-valued target (`fuzzy_target` = 0/1/2) üzerinde eğitilmektedir.
> Bu nedenle fuzzy modelde yüksek accuracy beklenen bir durumdur; bu sonuç modelin
> kural tabanlı fuzzy hedefi öğrenme başarısını gösterir.**
>
> Fuzzy hedef değişken doğrudan `cardio` sütunundan değil; `gender`, `age_years`
> ve `other_factors` üzerinden 12 karar kuralıyla türetilmektedir.

---

## 📎 Veri Kaynağı

Kaggle: [Cardiovascular Disease Dataset](https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset)

- Dosya adı genellikle: `cardio_train.csv`
- CSV ayraç desteği: `;` (varsayılan Kaggle formatı) veya `,` – **otomatik algılanır**

---

## ⚕️ Sorumluluk Reddi

Bu sistem **yalnızca eğitim ve araştırma amaçlıdır**. Tıbbi teşhis yerine geçmez.
Sağlık durumunuz için mutlaka bir sağlık uzmanına danışınız.
