# Bulanık Üç Değerli Mantık ile Hibrit Makine Öğrenmesi Modeli Geliştirme

## 1. Kapak Sayfası
* **Proje Adı:** Bulanık Üç Değerli Mantık ile Hibrit Makine Öğrenmesi Modeli Geliştirme
* **Ders Adı:** Makine Öğrenmesi
* **Konu:** Bulanık Üç Değerli Mantık ile Hibrit ML Modeli
* **Kullanılan Veri Seti:** Kaggle Cardiovascular Disease Dataset
* **Kullanılan Teknolojiler:** Python, Streamlit, SQLite, Scikit-learn
* **Hazırlayan:** [Adınız Soyadınız]

## 2. Özet
Bu projede, kalp hastalığı riskini tahmin etmek amacıyla geleneksel ikili (binary) sınıflandırma yöntemlerinin belirsiz durumları ifade etmekteki kısıtlılıkları ele alınmış ve bu kısıtlılıkları aşmak için Bulanık Üç Değerli Mantık (Fuzzy Three-Valued Logic) mimarisi kullanılarak hibrit bir makine öğrenmesi modeli geliştirilmiştir. Klasik binary sınıflandırma, hastaları yalnızca "Risk Var" (1) veya "Risk Yok" (0) şeklinde etiketlerken, bu çalışmada "Risk Olabilir" (0.5) şeklinde ara bir sınıf oluşturularak tıbbi karar alma sürecindeki belirsizliklerin daha iyi modellenebilmesi hedeflenmiştir. 

Geliştirilen bu hibrit makine öğrenmesi sisteminde, orijinal veri seti ile kurallara dayalı olarak oluşturulmuş "fuzzy-modified" veri seti üzerinde yedi farklı makine öğrenmesi algoritması eğitilmiştir. Orijinal ve bulanık tabanlı modellerin Accuracy, Precision, F1-Score ve hesaplama süresi (computation time) metrikleri bakımından karşılaştırması yapılmıştır. Ayrıca projeye, analiz süreçlerinin kullanıcı dostu bir şekilde yürütülmesini sağlamak amacıyla Streamlit tabanlı web arayüzü (dashboard) ve verilerin kalıcı bir biçimde yönetilebilmesi için SQLite veritabanı entegre edilmiştir.

## 3. Giriş
Kalp ve damar hastalıkları (kardiyovasküler hastalıklar), dünya genelinde en yaygın ölüm nedenlerinden biri olmaya devam etmektedir. Bu nedenle, kalp hastalığı riskinin bireyler üzerinde erken ve doğru bir şekilde tespit edilmesi büyük bir klinik önem taşımaktadır. Makine öğrenmesi algırmaları, tıbbi verilerden karmaşık örüntüleri öğrenerek hastalık teşhisine ve risk değerlendirmesine önemli katkılar sağlamaktadır. Ancak geleneksel makine öğrenmesi yaklaşımları genellikle sınıflandırma problemlerini ikili (binary) bir yapıda ele alır.

İkili sınıflandırmanın en büyük dezavantajı, hafif risk taşıyan, sınırda değerlere sahip olan veya tam olarak kesin tanı konulamayan "belirsiz" durumları ifade etmekte yetersiz kalmasıdır. Tıp biliminde kesin sınırlar nadirdir; hastalıklar aşamalı gelişim gösterir. "Risk Var" veya "Risk Yok" etiketleri, karar destek sistemlerinin hassasiyetini düşürür. Bu projede, "Risk Olabilir" (0.5) ara sınıflarının dahil edilmesi, gri alanları daha doğal bir şekilde modelleme imkanı tanıması açısından kritik bir çözüm olarak sunulmuştur.

## 4. Kullanılan Veri Seti
Projede Kaggle platformunda paylaşılan "Cardiovascular Disease Dataset" kullanılmıştır. Veri setindeki kolonlar ve açıklamaları aşağıdaki gibidir:

| Kolon | Açıklama |
|---|---|
| id | Hasta eşsiz numarası (eğitimde çıkarıldı) |
| age | Gün cinsinden yaş |
| gender | Cinsiyet (1: Kadın, 2: Erkek) |
| height | Boy (cm) |
| weight | Kilo (kg) |
| ap_hi | Sistolik kan basıncı |
| ap_lo | Diyastolik kan basıncı |
| cholesterol | Kolesterol (1: Normal, 2: Yüksek, 3: Çok Yüksek) |
| gluc | Glikoz (1: Normal, 2: Yüksek, 3: Çok Yüksek) |
| smoke | Sigara (0: Hayır, 1: Evet) |
| alco | Alkol (0: Hayır, 1: Evet) |
| active | Fiziksel aktivite (0: Hayır, 1: Evet) |
| cardio | Binary hedef değişken (0: Hastalık Yok, 1: Hastalık Var) |

## 5. Proje Mimarisi
Geliştirilen Hibrit Makine Öğrenmesi Modeli mimarisi şu şekildedir:
1. CSV veri seti otomatik ayraç algılama ile yüklenir.
2. Ham veri SQLite `raw_heart_data` tablosuna kaydedilir.
3. Yaş dönüşümü, BMI hesabı ve outlier temizliği uygulanır.
4. Bulanık mantık ile `other_factors` sütunu oluşturulur.
5. 12 karar kuralı ile 3 değerli bulanık risk etiketi (`fuzzy_target`) oluşturulur.
6. 7 ML algoritması Orijinal veri üzerinde eğitilir.
7. Aynı 7 ML algoritması Fuzzy veri üzerinde eğitilir.
8. Metrikler (Accuracy, F1, vb.) hesaplanıp karşılaştırılır.
9. Streamlit dashboard üzerinden grafikler ve metrikler görselleştirilir.
10. Kullanıcı yeni form değerleri girerek risk tahmini yapabilir.

## 6. Dosya ve Modül Açıklamaları
Aşağıda projenin dosya mimarisi detaylandırılmıştır:

| Dosya Adı | Görevi / İçeriği | Nerede Kullanıldığı |
|---|---|---|
| app.py | Streamlit dashboard ana giriş dosyası. Sayfa yapıları ve formları içerir. | Kullanıcı arayüzünün (UI) çalıştırılmasında |
| src/database.py | SQLite veritabanı yönetim sınıfını (DatabaseManager) barındırır. | Tablo oluşturma, veri kaydetme ve okumada |
| src/preprocessing.py | Veri temizleme, outlier tespiti, yaş ve BMI dönüşüm işlemlerini yapar. | Pipeline veri hazırlık aşamasında |
| src/fuzzy_logic.py | Bulanık mantıkla `other_factors` değerini hesaplayan kural fonksiyonlarıdır. | Yaşam tarzı değişkenlerinin birleştirilmesinde |
| src/rules.py | 12 karar kuralı üzerinden nihai fuzzy sınıfı hesaplayan motor. | Bulanık hedeflerin (`fuzzy_target`) üretilmesinde |
| src/train.py | 7 makine öğrenmesi modelini scikit-learn ile eğiten modül. | Modellerin Original/Fuzzy veri setlerinde eğitilmesinde |
| src/evaluate.py | Modellerin Accuracy, F1 vb. skorlarını ve Karmaşıklık matrislerini çıkarır. | Başarı değerlendirmesinde |
| src/visualization.py | Seaborn ve Matplotlib üzerinden model grafiklerini çizer. | Eğitim sonrasında görsellerin üretilmesinde |
| scripts/run_pipeline.py | Tüm işlemleri UI olmadan konsoldan (CLI) çalıştırır. | Arka planda test ve batch run işlemlerinde |

## 7. SQLite Veri Yönetimi
Verilerin kaybolmaması için 5 adet tablo kullanılmıştır:
* **raw_heart_data:** Yüklenen CSV'yi tutar (replace).
* **processed_original_data:** Temizlenmiş ve `cardio` hedefi içeren binary tablo (replace).
* **fuzzy_modified_data:** Bulanık kurallarla hesaplanmış `other_factors` ve `fuzzy_target` tablosu (replace).
* **model_results:** 14 modelin performans analiz sonuçlarını tutar (clear + append).
* **user_predictions:** Kullanıcıların dashboard üzerinden yaptığı tahmin geçmişi (append).

## 8. Veri Ön İşleme Aşamaları
Veri kalitesi için şu işlemler yürütülmüştür:
* **id:** Gürültü yaratmaması için çıkarılmıştır.
* **age:** Gün değerinden yıllara çevrilmiştir (`age / 365.25`).
* **bmi:** Vücut Kitle İndeksi `weight / (height/100)^2` ile hesaplanmıştır.
* **Aykırı Değer (Outlier) Temizliği:**
  * 20 <= Yaş <= 100
  * 10 <= BMI <= 70
  * 80 <= ap_hi <= 250
  * 40 <= ap_lo <= 150
Sınır dışı kalan hastalar çıkarılmış, modelin aşırı uçlardan etkilenmesi engellenmiştir.

## 9. Bulanık Üç Değerli Mantık Yaklaşımı
Fuzzy Three-Valued Logic 0, 0.5 ve 1 değerlerini kullanır. `active` değişkeni fiziksel aktiviteyi (iyi bir şey) ifade ettiği için `inactive = 1 - active` olarak terslenir.
Değişkenler: x=smoke, y=alco, z=inactive.

| smoke | alco | inactive | other_factors | Açıklama |
|---|---|---|---|---|
| 0 | 0 | 0 | 0.0 | Tüm faktörler sağlıklı (Risk Yok) |
| 1 | 1 | 1 | 1.0 | Tüm risk faktörleri mevcut (Yüksek Risk) |
| Diğer | Diğer | Diğer | 0.5 | Karma / belirsiz yaşam tarzı (Kısmi Risk) |

Bu sayede 3 değişken bir araya getirilerek modele daha açıklanabilir bir özellik (feature) sağlanır.

## 10. 12 Karar Kuralı
Yaş, Cinsiyet (Kadın eşik: 55, Erkek eşik: 45) ve Other Factors baz alınarak 12 adet karar kuralı tasarlanmıştır. Algoritmada Genç, Orta Yaş ve Yaşlı olarak optimize edilmiştir.

| Kural | Cinsiyet | Yaş | Other Factors | Sonuç | Model Encoding |
|---|---|---|---|---|---|
| 1 | Kadın | Genç (<40) | 0.0 | Risk Yok | 0 |
| 2-3 | Kadın | Genç (<40) | 0.5 - 1.0 | Risk Olabilir | 2 |
| 4-5 | Kadın | Orta Yaş | 0.0 - 0.5 | Risk Olabilir | 2 |
| 6 | Kadın | Orta Yaş | 1.0 | Risk Var | 1 |
| 7 | Erkek | Genç (<40) | 0.0 | Risk Yok | 0 |
| 8 | Erkek | Genç (<40) | 0.5 | Risk Olabilir | 2 |
| 9 | Erkek | Genç (<40) | 1.0 | Risk Var | 1 |
| 10 | Erkek | Orta Yaş | 0.0 | Risk Olabilir | 2 |
| 11 | Erkek | Orta Yaş | 0.5 - 1.0 | Risk Var | 1 |
| 12 | Herhangi | Yaşlı (>60) | Herhangi | Risk Var | 1 |

Encoding 0.5 için 2 atanarak Scikit-learn'in sınıf etiketini ayrık bir sınıf olarak işlemesi sağlanmıştır.

## 11. Kullanılan Algoritmalar
**1. GaussianNB:** Varsayılan değerlerle veri dağılımının normal olduğu varsayımıyla çalışan hızlı bir yöntemdir. Baseline oluşturmak için kullanılmıştır.
**2. SVM (Linear):** Sınıflar arası ayrımı maksimize eder. Ölçekten etkilendiği için StandardScaler Pipeline içinde `kernel="linear"` ile çalıştırılmıştır.
**3. AdaBoost:** Zayıf karar ağaçlarını hataları ağırlıklandırarak birleştirir. `n_estimators=100` kullanılarak seçilmiştir.
**4. Decision Tree:** Dallanma yapısıyla karar verir, overfitting'i önlemek için `max_depth=10` kısıtlaması eklenmiştir.
**5. KNN:** En yakın k adet komşuya bakar. `n_neighbors=15` ve StandardScaler ile eğitilmiştir.
**6. Random Forest:** Birçok ağacı bir araya getirerek genelleme sağlar. `n_estimators=25` kullanılmıştır.
**7. Gradient Boosting:** Ağaçları hataları minimize edecek şekilde ardışık bağlar, yüksek doğruluğu sebebiyle eklenmiştir.

## 12. Eğitim ve Test Süreci
Orijinal veri seti X özelliklerinde klinik ve fiziksel değerler alınarak hedef Y olarak `cardio` kullanılmıştır. Fuzzy veri setinde ise `age_years`, `gender` ve üretilmiş `other_factors` alınarak hedef Y olarak `fuzzy_target` (0,1,2) verilmiştir.
%80 eğitim, %20 test verisi ayrımı için `train_test_split` ve sınıfların dengeli dağılması için `stratify=y` kullanılmış; tüm modeller `.joblib` formatında diske kaydedilmiştir.

## 13. Değerlendirme Metrikleri
* **Accuracy:** Doğru tahminlerin genel oranıdır.
* **Precision:** Doğru pozitiflerin, tahmin edilen pozitiflere oranıdır.
* **F1 Macro:** Precision ve Recall'un dengesiz veriler için harmonik ortalamasıdır.
* **Computation Time:** Algoritmanın `fit()` işlemini tamamlama süresidir (sn).
* **Confusion Matrix:** Hatalı sınıflandırmaların nerede yaşandığını görselleştiren ısı haritası matrisidir.

## 14. Model Sonuçları
Veritabanı (`model_results`) analizine göre:
Orijinal (Binary) veri seti üzerindeki modeller ortalama %71 - %73 başarı aralığında kalmaktadır. Oysa Fuzzy-Modified (3 sınıflı) veri seti üzerinde modeller (örneğin Decision Tree, Gradient Boosting) %99 - %100 doğruluğa erişmektedir. SVM hesaplama süresi açısından 600+ saniye sürerken Naive Bayes anlık sonuç vermiştir.
* **Accuracy Gain:** Modellerin fuzzy üzerinde gösterdiği muazzam başarı artışını belirten `accuracy_gain.png` dosyasında da netçe görüleceği gibi, algoritmalar 12 karar kuralı yapısını ezberleme eğilimi göstermiştir.

## 15. Streamlit Dashboard
Web arayüzünde 4 sayfa yer alır:
* **Genel Bakış:** Proje özeti ve 3 renkli bulanık kart yapısını açıklar.
* **Veri Yükle & Pipeline:** CSV yüklenip sırayla veritabanı, preprocessing, fuzzy oluşturma ve eğitim adımlarının çalıştırıldığı sistem yönetim merkezidir.
* **Model Sonuçları:** En iyi iki modelin gösterildiği ve Matplotlib/Seaborn grafiklerinin (karşılaştırma, gain, matris) yer aldığı analiz sekmesidir.
* **Risk Tahmini:** Kullanıcıların form doldurarak hasta sonuçları alabildiği ve uyarı renklerinin gösterildiği son katmandır.

## 16. Risk Tahmini Modülü
Kullanıcıdan alınan demografik, klinik ve yaşam tarzı girdileri (yaş, boy, tansiyon, sigara vb.) üzerinden anlık olarak BMI ve `inactive` değişkenleri türetilir. Bu değişkenler yardımıyla `other_factors` hesaplanıp hastanın 12 karar kuralından hangisine tabi olduğu bulunur. Sonuç (Risk Yok, Olabilir, Var) büyük uyarı kartları ile ekrana yansıtılır ve `user_predictions` tablosuna hasta tahmini olarak kaydedilir.

## 17. Akademik Değerlendirme
Orijinal modelin klinik teşhis üzerindeki karmaşık ikili tahmini yerine, Fuzzy-Modified model deterministik olarak oluşturulmuş 12 karar kuralının sonucunu öğrenmektedir. **Bu sebeple fuzzy modelde %100'e varan yüksek Accuracy çıkması beklenen bir durumdur.** Bu sonuç klinik bir mucize değil, kural tabanlı hedefi öğrenme yeteneğinin (pattern-recognition) mükemmelliğidir. Bulanık 0.5 sınıfı, klinik doktorları için faydalı bir erken uyarı/koruyucu sağlık bölgesi oluşturmaktadır. Sistem tıbbi teşhis yerine geçmez.

## 18. Avantajlar ve Sınırlılıklar
* **Avantajlar:** Binary'ye kıyasla daha esnek ve açıklanabilir sonuçlar verir. Belirsiz durumları modelleyebilir. SQLite ve Streamlit entegrasyonu sayesinde uçtan uca modern bir veri bilimi projesidir.
* **Sınırlılıklar:** Fuzzy hedef tamamen manuel kurallardan üretildiği için algoritmaların yüksek accuracy vermesi kaçınılmazdır. Sistem gerçek uzman doktorlar tarafından doğrulanan klinik etiketlerden yoksundur.

## 19. Sonuç
Bu projede, Bulanık Üç Değerli Mantık kullanılarak 12 kural çerçevesinde "other_factors" sentezlenmiş ve 3 sınıflı bir risk yapısı kurulmuştur. Orijinal ve Fuzzy veriler üzerinde 7 makine öğrenmesi modeli başarıyla karşılaştırılmış; sistem Streamlit ile modern bir arayüze taşınırken SQLite ile kalıcı hale getirilmiştir. Proje; yapay zeka, veri bilimi, bulanık mantık ve web programlamayı birleştiren tam donanımlı bir hibrit akademik çalışma olmuştur.

## 20. Kaynakça
1. A novel fuzzy three-valued logic computational framework in machine learning for medicine dataset
2. Kaggle Cardiovascular Disease Dataset
3. Scikit-learn, Streamlit, SQLite official documentations
