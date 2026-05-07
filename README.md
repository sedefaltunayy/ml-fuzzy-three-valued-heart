# HibritML – Bulanık Üç Değerli Mantık ile Hibrit ML Modeli

## 📋 Proje Özeti

Bu proje, klasik binary (0-1) sınıflandırmanın belirsiz risk durumlarını temsil etmekte yetersiz kalması probleminden yola çıkılarak geliştirilmiştir. Geleneksel makine öğrenmesi modelleri çoğunlukla bir kişiyi yalnızca **Risk Yok** veya **Risk Var** şeklinde sınıflandırır. Ancak kardiyovasküler risk tahmini gibi alanlarda bazı bireyler kesin riskli ya da kesin risksiz olmayabilir. Bu nedenle projede ara bir durum olan **Risk Olabilir** sınıfı da modellenmiştir.

Projede Kaggle Cardiovascular Disease Dataset kullanılmıştır. Veri seti üzerinde ön işleme adımları uygulanmış, yaşam tarzı faktörlerinden `other_factors` isimli bulanık özellik üretilmiş ve ardından 7 farklı makine öğrenmesi algoritması hem original veri seti hem de fuzzy-modified veri seti üzerinde eğitilmiştir.

| Katman | Açıklama |
|---|---|
| **Girdi** | Kaggle Cardiovascular Disease Dataset (`;` veya `,` ayıracı otomatik algılanır) |
| **Preprocessing** | Yaş dönüşümü, BMI hesabı, eksik değer yönetimi, duplicate temizliği ve tıbbi outlier temizliği |
| **Fuzzy Girdi** | `other_factors = f(smoke, alco, active)` ve değer kümesi `{0.0, 0.5, 1.0}` |
| **Fuzzy Hedef** | `gender + age_years + other_factors` üzerinden 12 karar kuralıyla `fuzzy_target` üretilir |
| **ML Katmanı** | 7 algoritma × 2 veri seti = toplam 14 model eğitimi |
| **Çıktı** | Accuracy, Precision, Recall, F1 Score, Computation Time, grafikler ve SQLite kayıtları |

---

## 📁 Dosya Yapısı

```text
ml-fuzzy-three-valued-heart/
├── app.py                        # Streamlit dashboard ana giriş dosyası
├── requirements.txt              # Proje bağımlılıkları
├── README.md
├── src/
│   ├── __init__.py               # Paket başlatıcı
│   ├── database.py               # SQLite bağlantı ve tablo yönetimi
│   ├── preprocessing.py          # CSV okuma, temizlik, BMI, outlier, delimiter detection
│   ├── fuzzy_logic.py            # other_factors hesabı
│   ├── rules.py                  # 12 karar kuralı motoru
│   ├── train.py                  # 7 model eğitimi ve joblib kayıt işlemleri
│   ├── evaluate.py               # Metrik hesaplama, classification report, confusion matrix
│   ├── ui_components.py          # Streamlit özel UI bileşenleri
│   └── visualization.py          # Grafik üretim fonksiyonları
└── scripts/
    └── run_pipeline.py           # CLI üzerinden tam pipeline çalıştırma
```

Çalışma zamanında otomatik oluşturulan dizinler:

```text
data/
database/
models/original/
models/fuzzy/
outputs/figures/
outputs/reports/
outputs/metrics/
```

---

## 🛠️ Kurulum

Projeyi klonladıktan sonra proje klasörüne girin:

```bash
git clone https://github.com/sedefaltunayy/ml-fuzzy-three-valued-heart.git
cd ml-fuzzy-three-valued-heart
```

Gerekli bağımlılıkları yükleyin:

```bash
pip install -r requirements.txt
```

Python 3.10 veya üzeri önerilir.

---

## 🚀 Çalıştırma

### Yöntem A – Streamlit Dashboard

```bash
streamlit run app.py
```

Dashboard üzerinden izlenecek temel adımlar:

1. **Veri Yükle & Pipeline** sayfasına git.
2. Kaggle `cardio_train.csv` dosyasını yükle.
3. SQLite veritabanını oluştur.
4. Preprocessing işlemini çalıştır.
5. Fuzzy dataset oluştur.
6. Tüm modelleri eğit.
7. **Model Sonuçları** sayfasında grafikleri incele.
8. **Risk Tahmini** sayfasında kullanıcı bilgileriyle tahmin yap.

---

### Yöntem B – CLI Pipeline

Streamlit arayüzü kullanılmadan tüm pipeline tek komutla çalıştırılabilir:

```bash
python scripts/run_pipeline.py --csv data/cardio_train.csv
```

Bu komut sırasıyla:

1. Gerekli klasörleri oluşturur.
2. SQLite tablolarını oluşturur.
3. CSV dosyasını okur.
4. Ham veriyi `raw_heart_data` tablosuna kaydeder.
5. Preprocessing uygular.
6. İşlenmiş veriyi `processed_original_data` tablosuna kaydeder.
7. Fuzzy dönüşümü uygular.
8. Fuzzy veriyi `fuzzy_modified_data` tablosuna kaydeder.
9. Original ve fuzzy veri setleri üzerinde modelleri eğitir.
10. Metrikleri, modelleri ve grafikleri çıktı klasörlerine kaydeder.

---

## 📎 Veri Seti

Projede kullanılan veri seti:

**Kaggle Cardiovascular Disease Dataset**

Beklenen dosya adı:

```text
cardio_train.csv
```

Dosya şu klasöre yerleştirilebilir:

```text
data/cardio_train.csv
```

Veri setindeki temel kolonlar:

| Kolon | Açıklama |
|---|---|
| `id` | Hasta kayıt numarası |
| `age` | Gün cinsinden yaş |
| `gender` | Cinsiyet |
| `height` | Boy |
| `weight` | Kilo |
| `ap_hi` | Sistolik kan basıncı |
| `ap_lo` | Diyastolik kan basıncı |
| `cholesterol` | Kolesterol seviyesi |
| `gluc` | Glikoz seviyesi |
| `smoke` | Sigara kullanımı |
| `alco` | Alkol kullanımı |
| `active` | Fiziksel aktivite durumu |
| `cardio` | Orijinal binary hedef değişken |

CSV ayıracı `;` veya `,` olabilir. Projede CSV okuma fonksiyonu bu ayracı otomatik algılayacak şekilde tasarlanmıştır.

---

## 🗄️ Veritabanı Şeması

Veritabanı dosyası:

```text
database/heart_disease.db
```

Projede kullanılan SQLite tabloları:

| Tablo | Açıklama | Yazma Stratejisi |
|---|---|---|
| `raw_heart_data` | Yüklenen ham CSV verisini saklar. | replace |
| `processed_original_data` | Ön işleme uygulanmış binary hedefli veriyi saklar. | replace |
| `fuzzy_modified_data` | `other_factors` ve `fuzzy_target` içeren fuzzy veri setini saklar. | replace |
| `model_results` | Eğitilen modellerin metrik sonuçlarını saklar. | clear + append |
| `user_predictions` | Dashboard üzerinden yapılan kullanıcı tahminlerini saklar. | append |

---

## ⚙️ Veri Ön İşleme

Preprocessing aşamasında aşağıdaki işlemler uygulanır:

1. CSV dosyası otomatik delimiter detection ile okunur.
2. Kolon adları standartlaştırılır.
3. `id` veya `record_id` gibi kayıt numarası kolonları çıkarılır.
4. `age` değeri gün cinsindeyse yıl cinsine dönüştürülür:

```text
age_years = age / 365.25
```

5. BMI hesaplanır:

```text
bmi = weight / ((height / 100) ** 2)
```

6. Tıbbi outlier sınırları uygulanır.
7. Eksik değerler medyan veya mod değerleriyle doldurulur.
8. Duplicate satırlar temizlenir.
9. Model eğitiminde kullanılacak kolonlar seçilir.

Projede kullanılan tıbbi outlier sınırları:

| Değişken | Alt Sınır | Üst Sınır |
|---|---:|---:|
| `ap_hi` | 80 | 250 |
| `ap_lo` | 50 | 150 |
| `height` | 100 | 250 |
| `weight` | 30 | 250 |
| `bmi` | 10 | 70 |

---

## 🧠 Bulanık Üç Değerli Mantık

Bu projede bulanık mantık iki seviyede kullanılmıştır:

1. Girdi seviyesinde `other_factors` üretimi
2. Hedef seviyesinde `fuzzy_target` üretimi

---

## 🔹 Other Factors Hesabı

`other_factors`, üç yaşam tarzı değişkeninden türetilir:

- `smoke`
- `alco`
- `active`

Projede uygulanan mantık:

| Durum | `other_factors` | Açıklama |
|---|---:|---|
| Sigara yok, alkol yok, fiziksel olarak aktif | 0.0 | Sağlıklı yaşam tarzı |
| Sigara var, alkol var, fiziksel olarak inaktif | 1.0 | Riskli yaşam tarzı |
| Diğer tüm kombinasyonlar | 0.5 | Karma / belirsiz yaşam tarzı |

Bu dönüşüm sayesinde üç ayrı binary yaşam tarzı değişkeni tek bir açıklanabilir fuzzy değişkene dönüştürülür.

---

## 🧠 Fuzzy Target Encoding

Projede fuzzy hedef değişkeni `fuzzy_target` olarak adlandırılmıştır.

**Projede kullanılan doğru sınıf eşleşmesi aşağıdaki gibidir:**

| `fuzzy_target` | Anlam | Renk |
|---:|---|---|
| 0 | Risk Yok | Yeşil |
| 1 | Risk Olabilir | Turuncu |
| 2 | Risk Var | Kırmızı |

> Not: `0.5` değeri, `other_factors` değişkeninde kullanılan bulanık ara durumdur.  
> Model hedef sınıfında ise ara risk durumu integer olarak `1 = Risk Olabilir` etiketiyle temsil edilir.

---

## 🧠 12 Karar Kuralı

Fuzzy hedef değişkeni, yaş grubu, cinsiyet ve `other_factors` değişkenine göre oluşturulur.

Projede kullanılan yaş grupları:

| Yaş Aralığı | Grup |
|---|---|
| `< 40` | Genç |
| `40–60` | Orta yaş |
| `> 60` | Yaşlı |

Kural tablosu:

| Kural | Cinsiyet | Yaş Grubu | Other Factors | Sonuç |
|---:|---|---|---:|---|
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
| 11 | Erkek | 40–60 | 0.5 veya 1.0 | 2 – Risk Var |
| 12 | Kadın / Erkek | > 60 | Herhangi | 2 – Risk Var |

---

## 📊 Model Eğitiminde Kullanılan Özellikler

### Original Dataset

Original veri setinde hedef değişken olarak `cardio` kullanılır.

Kullanılan özellikler:

```text
age_years
gender
height
weight
bmi
ap_hi
ap_lo
cholesterol
gluc
smoke
alco
active
```

Hedef değişken:

```text
cardio
```

Bu yapı klasik binary classification yaklaşımını temsil eder.

---

### Fuzzy-Modified Dataset

Fuzzy-modified veri setinde hedef değişken olarak `fuzzy_target` kullanılır.

Kullanılan özellikler:

```text
age_years
gender
height
weight
bmi
ap_hi
ap_lo
cholesterol
gluc
other_factors
```

Hedef değişken:

```text
fuzzy_target
```

Bu yapı bulanık mantık destekli üç sınıflı hibrit sınıflandırma yaklaşımını temsil eder.

---

## 🤖 Kullanılan Makine Öğrenmesi Algoritmaları

Projede 7 farklı makine öğrenmesi algoritması kullanılmıştır.

| Model | Parametre / Açıklama |
|---|---|
| GaussianNB | Varsayılan parametreler |
| SVM Linear | `StandardScaler` + `SVC(kernel="linear")` |
| AdaBoost | `n_estimators=100`, `algorithm="SAMME"` |
| Decision Tree | `max_depth=10` |
| KNN | `StandardScaler` + `n_neighbors=15` |
| Random Forest | `n_estimators=25`, `max_features=min(4, n_features)` |
| Gradient Boosting | `n_estimators=90` |

SVM ve KNN algoritmaları ölçekten etkilendiği için `StandardScaler` içeren `Pipeline` yapısı ile kullanılmıştır.

---

## 🧪 Eğitim Süreci

Her iki veri seti üzerinde aynı 7 algoritma eğitilir:

```text
7 algoritma × 2 veri seti = 14 model
```

Veri bölme işlemi:

- Eğitim oranı: %80
- Test oranı: %20
- `random_state=42`
- Mümkün olduğunda `stratify=y`

Eğitilen modeller `joblib` formatında kaydedilir:

```text
models/original/{model_name}.joblib
models/fuzzy/{model_name}.joblib
```

---

## 📈 Değerlendirme Metrikleri

Modeller aşağıdaki metriklerle değerlendirilir:

| Metrik | Açıklama |
|---|---|
| Accuracy | Doğru tahminlerin tüm tahminlere oranı |
| Precision | Pozitif tahminlerin ne kadarının doğru olduğunu gösterir |
| Recall | Gerçek pozitiflerin ne kadarının doğru yakalandığını gösterir |
| F1 Score | Precision ve Recall metriklerinin harmonik ortalaması |
| Computation Time | Modelin eğitim süresini gösterir |
| Classification Report | Sınıf bazlı precision, recall ve F1 skorlarını verir |
| Confusion Matrix | Sınıflandırma hatalarının hangi sınıflarda oluştuğunu gösterir |

---

## 📂 Üretilen Çıktılar

Model eğitimi tamamlandığında aşağıdaki dosyalar oluşturulur:

```text
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

## 🖥️ Streamlit Dashboard

Dashboard dört ana bölümden oluşur:

### Genel Bakış

Proje amacı, fuzzy logic yaklaşımı, kullanılan algoritmalar ve genel sistem mimarisi açıklanır.

### Veri Yükle & Pipeline

Bu bölümde kullanıcı CSV dosyasını yükleyebilir ve pipeline adımlarını çalıştırabilir:

1. SQLite veritabanı oluşturma
2. Preprocessing çalıştırma
3. Fuzzy dataset oluşturma
4. Tüm modelleri eğitme

### Model Sonuçları

Eğitilen modellerin sonuçları karşılaştırmalı olarak gösterilir:

- Accuracy karşılaştırması
- Precision karşılaştırması
- Computation time karşılaştırması
- Accuracy gain grafiği
- Confusion matrix çıktıları

### Risk Tahmini

Kullanıcıdan alınan bilgilerle risk tahmini yapılır:

- Yaş
- Cinsiyet
- Sigara kullanımı
- Alkol kullanımı
- Fiziksel aktivite durumu

Sistem bu değerlerden `other_factors` değerini hesaplar ve 12 kural motoruyla sonucu üretir.

---

## ⚠️ Önemli Akademik Not

Original model, veri setinin kendi binary hedef değişkeni olan `cardio` üzerinde eğitilir. Fuzzy-modified model ise `gender`, `age_years` ve `other_factors` üzerinden üretilen kural tabanlı `fuzzy_target` üzerinde eğitilir.

Bu nedenle fuzzy modelde yüksek accuracy elde edilmesi beklenen bir durumdur. Bu sonuç doğrudan klinik teşhis başarısı olarak yorumlanmamalıdır. Yüksek başarı, modelin oluşturulan kural tabanlı fuzzy hedef yapısını başarılı biçimde öğrendiğini gösterir.

Bu sistem eğitim ve araştırma amaçlıdır. Tıbbi teşhis yerine geçmez.

---

## 📎 Veri Kaynağı

Kaggle Cardiovascular Disease Dataset:

https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset

Dosya adı genellikle:

```text
cardio_train.csv
```

---

## ⚕️ Sorumluluk Reddi

Bu sistem yalnızca eğitim ve araştırma amaçlı geliştirilmiştir. Herhangi bir tıbbi teşhis, tedavi veya klinik karar verme amacıyla kullanılmamalıdır. Sağlık durumunuzla ilgili kararlar için mutlaka bir sağlık uzmanına danışınız.