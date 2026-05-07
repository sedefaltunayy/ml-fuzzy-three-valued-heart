"""
app.py – Streamlit Dashboard (HibritML)
"""
import os, sys, io, warnings
import pandas as pd
import numpy as np
import streamlit as st

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.database      import DatabaseManager
from src.preprocessing import load_csv, preprocess
from src.fuzzy_logic   import apply_fuzzy_other_factors, get_other_factors_from_values, describe_other_factors
from src.rules         import apply_rules_to_dataframe, predict_single, FUZZY_LABELS, FUZZY_COLORS
from src.train         import run_full_training, ensure_dirs, list_trained_models
from src.visualization import generate_all_figures
from src.evaluate      import evaluate_model, print_evaluation_summary
from src.ui_components import (
    inject_custom_css, render_sidebar, render_hero_header,
    render_metric_card, render_pipeline_step,
    render_risk_result_card, render_footer_warning
)

# ── Klasörler ──────────────────────────────────────────────────────────────
for d in ["data","database","models/original","models/fuzzy",
          "outputs/figures","outputs/reports","outputs/metrics"]:
    os.makedirs(d, exist_ok=True)

DB = DatabaseManager()

# ── Sayfa yapılandırması ───────────────────────────────────────────────────
st.set_page_config(
    page_title="HibritML – Kardiyovasküler Risk Analizi",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# UI/UX CSS Injections
inject_custom_css()

# ── Sidebar Navigasyon ─────────────────────────────────────────────────────
page = render_sidebar()

# ── Header (Tüm sayfalarda) ────────────────────────────────────────────────
render_hero_header()

# ══════════════════════════════════════════════════════════════════════════
# SAYFA 1 – GENEL BAKIŞ
# ══════════════════════════════════════════════════════════════════════════
if page == "🏠 Genel Bakış":
    st.markdown('<div class="modern-card">', unsafe_allow_html=True)
    st.markdown("<h3>🎯 Proje Amacı</h3>", unsafe_allow_html=True)
    st.write("Klasik binary sınıflandırmanın belirsiz durumları temsil edememesi sorununa bulanık mantık tabanlı bir çözüm sunar. `smoke`, `alco`, `active` değişkenleri birleştirilerek `other_factors` (0, 0.5, 1) üretilir; 12 karar kuralıyla hedef 3 sınıfa dönüştürülerek risk tahmini iyileştirilir.")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<h3>🧠 Fuzzy Three-Valued Logic Sınıfları</h3>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
        <div class="modern-card" style="border-top: 4px solid #2ecc71; text-align: center;">
            <h1 style="color:#2ecc71; margin:0;">0</h1>
            <h4>Risk Yok</h4>
            <p style="color:#94a3b8; font-size:0.9rem;">Sağlıklı durum</p>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="modern-card" style="border-top: 4px solid #f39c12; text-align: center;">
            <h1 style="color:#f39c12; margin:0;">0.5</h1>
            <h4>Risk Olabilir</h4>
            <p style="color:#94a3b8; font-size:0.9rem;">Belirsiz durum</p>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div class="modern-card" style="border-top: 4px solid #e74c3c; text-align: center;">
            <h1 style="color:#e74c3c; margin:0;">1</h1>
            <h4>Risk Var</h4>
            <p style="color:#94a3b8; font-size:0.9rem;">Hastalık riski yüksek</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<h3>⚙️ Kullanılan ML Algoritmaları (7 Adet)</h3>", unsafe_allow_html=True)
    models = ["GaussianNB", "SVM (Linear)", "AdaBoost", "DecisionTree", "KNN", "RandomForest", "GradientBoosting"]
    badges = "".join([f'<span class="badge badge-blue">{m}</span>' for m in models])
    st.markdown(f'<div style="margin-bottom: 20px;">{badges}</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="modern-card">
        <h3>🔍 Original vs Fuzzy Hybrid Karşılaştırması</h3>
        <p>Proje kapsamında iki farklı eğitim seti kullanılır ve 14 model eğitilir:</p>
        <ul>
            <li><b>Original Dataset:</b> Geleneksel Binary Target (0 veya 1).</li>
            <li><b>Fuzzy Dataset:</b> Kural tabanlı üretilmiş 3 değerli (0, 1, 2) hedef. Modelin kuralları öğrenme yeteneği test edilir.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════
# SAYFA 2 – VERİ YÜKLE & PİPELİNE
# ══════════════════════════════════════════════════════════════════════════
elif page == "📂 Veri Yükle & Pipeline":
    st.markdown("<h3>1️⃣ CSV Dosyası Yükle</h3>", unsafe_allow_html=True)
    
    with st.container():
        st.markdown('<div class="modern-card">', unsafe_allow_html=True)
        uploaded = st.file_uploader("Kaggle Cardiovascular Disease CSV", type=["csv"])
        
        if uploaded:
            raw_bytes = uploaded.read()
            loaded_df = None
            for sep in [";", ","]:
                try:
                    tmp = pd.read_csv(io.BytesIO(raw_bytes), sep=sep)
                    if tmp.shape[1] >= 5:
                        loaded_df = tmp
                        break
                except Exception:
                    continue
            if loaded_df is None:
                try:
                    loaded_df = pd.read_csv(io.BytesIO(raw_bytes), sep=None, engine="python")
                except Exception as e:
                    st.error(f"CSV okunamadı: {e}")

            if loaded_df is not None:
                st.session_state["raw_df"] = loaded_df
                st.success(f"✅ CSV başarıyla yüklendi: {loaded_df.shape[0]:,} satır")
                with st.expander("Veri Önizleme"):
                    st.dataframe(loaded_df.head(3), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<h3>2️⃣ Pipeline Adımları</h3>", unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)

    # State checks
    has_raw = "raw_df" in st.session_state
    has_proc = "processed_df" in st.session_state
    has_fuzzy = "fuzzy_df" in st.session_state

    # Card 1
    with col1:
        render_pipeline_step("🗄️", "Veritabanı Oluştur", "Bekliyor" if not has_raw else "Hazır", "blue" if has_raw else "orange")
        if st.button("SQLite Oluştur", key="btn_db"):
            DB.create_tables()
            st.success("Tablolar oluşturuldu!")
            
    # Card 2
    with col2:
        render_pipeline_step("⚙️", "Preprocessing", "Tamamlandı" if has_proc else "Bekliyor", "green" if has_proc else "orange")
        if st.button("Preprocessing", key="btn_prep", disabled=not has_raw):
            with st.spinner("İşleniyor..."):
                proc = preprocess(st.session_state["raw_df"])
                DB.to_sql(st.session_state["raw_df"], "raw_heart_data", if_exists="replace")
                DB.to_sql(proc, "processed_original_data", if_exists="replace")
                st.session_state["processed_df"] = proc
                st.success("Tamamlandı!")
                st.rerun()

    # Card 3
    with col3:
        render_pipeline_step("🌀", "Fuzzy Dataset", "Tamamlandı" if has_fuzzy else "Bekliyor", "green" if has_fuzzy else "orange")
        if st.button("Fuzzy Oluştur", key="btn_fuzzy", disabled=not has_proc):
            with st.spinner("Bulanıklaştırılıyor..."):
                proc = st.session_state["processed_df"]
                fuzzy = apply_fuzzy_other_factors(proc)
                fuzzy = apply_rules_to_dataframe(fuzzy)
                fuzzy_cols = ["age_years","gender","height","weight","bmi","ap_hi","ap_lo","cholesterol","gluc","other_factors","fuzzy_target"]
                fuzzy_save = fuzzy[[c for c in fuzzy_cols if c in fuzzy.columns]]
                DB.to_sql(fuzzy_save, "fuzzy_modified_data", if_exists="replace")
                st.session_state["fuzzy_df"] = fuzzy
                st.success("Oluşturuldu!")
                st.rerun()

    # Card 4
    with col4:
        render_pipeline_step("🤖", "Model Eğitimi", "Hazır" if has_fuzzy else "Bekliyor", "blue" if has_fuzzy else "orange")
        if st.button("Tüm Modelleri Eğit", key="btn_train", disabled=not has_fuzzy):
            with st.spinner("14 model eğitiliyor: 7 original + 7 fuzzy... Lütfen bekleyin."):
                ensure_dirs()
                metrics_df = run_full_training(st.session_state["processed_df"], st.session_state["fuzzy_df"])
                DB.clear_table("model_results")
                DB.to_sql(metrics_df, "model_results", if_exists="append")
                generate_all_figures(metrics_df, st.session_state["processed_df"], st.session_state["fuzzy_df"])
                st.session_state["metrics_df"] = metrics_df
                st.success("Eğitim Tamamlandı!")

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<h3>📋 Veritabanı Durumu</h3>", unsafe_allow_html=True)
    
    m1, m2, m3, m4, m5 = st.columns(5)
    with m1: render_metric_card("📂", "Ham Veri", f"{DB.row_count('raw_heart_data'):,}", "blue")
    with m2: render_metric_card("⚙️", "İşlenmiş Veri", f"{DB.row_count('processed_original_data'):,}", "green")
    with m3: render_metric_card("🌀", "Fuzzy Veri", f"{DB.row_count('fuzzy_modified_data'):,}", "orange")
    with m4: render_metric_card("🤖", "Model Sonuçları", f"{DB.row_count('model_results'):,}", "blue")
    with m5: render_metric_card("👤", "Kullanıcı Tahmin", f"{DB.row_count('user_predictions'):,}", "green")

# ══════════════════════════════════════════════════════════════════════════
# SAYFA 3 – MODEL SONUÇLARI
# ══════════════════════════════════════════════════════════════════════════
elif page == "📊 Model Sonuçları":
    metrics_df = None
    if "metrics_df" in st.session_state:
        metrics_df = st.session_state["metrics_df"]
    elif DB.row_count("model_results") > 0:
        metrics_df = DB.read_sql("model_results")

    if metrics_df is None or metrics_df.empty:
        st.info("Henüz model eğitimi yapılmamış. 'Veri Yükle & Pipeline' sayfasından modelleri eğitin.")
        st.stop()

    orig = metrics_df[metrics_df["dataset_type"]=="original"]
    fuzz = metrics_df[metrics_df["dataset_type"]=="fuzzy"]

    c1, c2, c3, c4 = st.columns(4)
    if not orig.empty:
        best_orig = orig.loc[orig["accuracy"].idxmax()]
        with c1:
            st.markdown(f"""
            <div class="modern-card" style="border-top: 4px solid #4A90D9;">
                <h4>🏆 En İyi Original</h4>
                <h2 style="color:#4A90D9; margin: 5px 0;">{best_orig['model_name']}</h2>
                <span class="badge badge-blue">Acc: {best_orig['accuracy']:.4f}</span>
            </div>
            """, unsafe_allow_html=True)
            
    if not fuzz.empty:
        best_fuzz = fuzz.loc[fuzz["accuracy"].idxmax()]
        with c2:
            st.markdown(f"""
            <div class="modern-card" style="border-top: 4px solid #E67E22;">
                <h4>🏆 En İyi Fuzzy</h4>
                <h2 style="color:#E67E22; margin: 5px 0;">{best_fuzz['model_name']}</h2>
                <span class="badge badge-orange">Acc: {best_fuzz['accuracy']:.4f}</span>
            </div>
            """, unsafe_allow_html=True)
            
    # Accuracy Gain
    if not orig.empty and not fuzz.empty:
        avg_gain = (fuzz["accuracy"].mean() - orig["accuracy"].mean()) * 100
        with c3:
            st.markdown(f"""
            <div class="modern-card" style="border-top: 4px solid #2ecc71;">
                <h4>📈 Avg Accuracy Gain</h4>
                <h2 style="color:#2ecc71; margin: 5px 0;">+{avg_gain:.2f}%</h2>
                <span class="badge badge-green">Fuzzy over Original</span>
            </div>
            """, unsafe_allow_html=True)
            
    # Avg Comp Time
    avg_time = metrics_df["computation_time"].mean()
    with c4:
        st.markdown(f"""
        <div class="modern-card" style="border-top: 4px solid #9b59b6;">
            <h4>⏱️ Avg Eğitim Süresi</h4>
            <h2 style="color:#9b59b6; margin: 5px 0;">{avg_time:.3f}s</h2>
            <span class="badge badge-blue">Per Model</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<h3>📈 Görselleştirmeler</h3>", unsafe_allow_html=True)
    st.markdown('<div class="modern-card">', unsafe_allow_html=True)
    
    fig_files = {
        "Accuracy Karşılaştırma":  "outputs/figures/accuracy_comparison.png",
        "Accuracy Gain":           "outputs/figures/accuracy_gain.png",
        "Precision Karşılaştırma": "outputs/figures/precision_comparison.png",
        "Hesaplama Süresi":        "outputs/figures/computation_time_comparison.png",
        "Sınıf Dağılımı – Orig":  "outputs/figures/class_distribution_original.png",
        "Sınıf Dağılımı – Fuzzy": "outputs/figures/class_distribution_fuzzy.png",
        "Confusion Matrix – Orig": "outputs/figures/confusion_matrix_original_best.png",
        "Confusion Matrix – Fuzzy":"outputs/figures/confusion_matrix_fuzzy_best.png",
        "Korelasyon Haritası":     "outputs/figures/feature_correlation.png",
    }

    tabs = st.tabs(list(fig_files.keys()))
    for tab, (label, path) in zip(tabs, fig_files.items()):
        with tab:
            if os.path.exists(path):
                st.image(path, use_container_width=True)
            else:
                st.info(f"Görsel henüz üretilmedi: {path}")
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("<h3>📋 Detaylı Sonuç Tablosu</h3>", unsafe_allow_html=True)
    styled = metrics_df.sort_values(["dataset_type","accuracy"], ascending=[True,False])
    st.dataframe(styled.round(4), use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════════════
# SAYFA 4 – RİSK TAHMİNİ
# ══════════════════════════════════════════════════════════════════════════
elif page == "🔬 Risk Tahmini":
    c_left, c_right = st.columns([1.2, 1])

    with c_left:
        st.markdown('<div class="modern-card">', unsafe_allow_html=True)
        st.markdown("<h3>Hasta Bilgileri Girişi</h3>", unsafe_allow_html=True)
        with st.form("risk_form"):
            c1, c2 = st.columns(2)
            with c1:
                gender = st.selectbox("Cinsiyet", [1, 2], format_func=lambda x: "Kadın" if x==1 else "Erkek")
                height = st.slider("Boy (cm)", 140, 210, 170)
                ap_hi = st.slider("Sistolik Kan Basıncı (mmHg)", 80, 250, 120)
                cholesterol = st.selectbox("Kolesterol", [1,2,3], format_func=lambda x: {1:"Normal",2:"Yüksek",3:"Çok Yüksek"}[x])
                smoke  = st.selectbox("Sigara Kullanımı", [0,1], format_func=lambda x: "Hayır" if x==0 else "Evet")
                active = st.selectbox("Fiziksel Aktivite", [0,1], format_func=lambda x: "Hayır" if x==0 else "Evet")
            with c2:
                age_years = st.slider("Yaş", 18, 90, 45)
                weight = st.slider("Kilo (kg)", 40, 180, 75)
                ap_lo = st.slider("Diastolik Kan Basıncı (mmHg)", 50, 150, 80)
                gluc = st.selectbox("Glukoz", [1,2,3], format_func=lambda x: {1:"Normal",2:"Yüksek",3:"Çok Yüksek"}[x])
                alco   = st.selectbox("Alkol Kullanımı", [0,1], format_func=lambda x: "Hayır" if x==0 else "Evet")
                bmi = round(weight / ((height/100)**2), 1)
                st.markdown(f"<div style='margin-top:30px;'><span class='badge badge-blue'>Hesaplanan BMI: {bmi}</span></div>", unsafe_allow_html=True)

            submitted = st.form_submit_button("🔍 Risk Analizi Yap", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with c_right:
        if submitted:
            other_factors = get_other_factors_from_values(smoke, alco, active)
            result = predict_single(gender, float(age_years), other_factors)
            
            render_risk_result_card(
                label=result["label"],
                target_value=result["fuzzy_target"],
                other_factors=other_factors,
                rule_desc=result["rules_applied"]
            )
            
            record = {
                "age_years": age_years, "gender": gender,
                "height": height, "weight": weight, "bmi": bmi,
                "ap_hi": ap_hi, "ap_lo": ap_lo,
                "cholesterol": cholesterol, "gluc": gluc,
                "smoke": smoke, "alco": alco, "active": active,
                "other_factors": other_factors,
                "fuzzy_target": result["fuzzy_target"],
                "fuzzy_label": result["label"],
            }
            try:
                DB.insert_prediction(record)
            except Exception:
                pass

        render_footer_warning()
        
        if DB.row_count("user_predictions") > 0:
            with st.expander("📋 Son Tahmin Geçmişi"):
                hist = DB.read_sql("user_predictions")
                if "predicted_at" in hist.columns:
                    hist = hist.sort_values("predicted_at", ascending=False).head(5)
                display_cols = [c for c in ["age_years","gender","fuzzy_target","fuzzy_label"] if c in hist.columns]
                st.dataframe(hist[display_cols], use_container_width=True, hide_index=True)
