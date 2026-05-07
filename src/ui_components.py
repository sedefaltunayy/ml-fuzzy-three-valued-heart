import streamlit as st

def inject_custom_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .stApp {
        background-color: #0b0f19;
    }
    
    /* Modern Cards (Glassmorphism) */
    .modern-card {
        background: linear-gradient(145deg, rgba(30, 34, 45, 0.7), rgba(22, 26, 35, 0.7));
        backdrop-filter: blur(10px);
        border-radius: 16px;
        padding: 20px;
        border: 1px solid rgba(255, 255, 255, 0.05);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.2);
        margin-bottom: 20px;
        transition: transform 0.25s ease, box-shadow 0.25s ease;
    }
    .modern-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 40px 0 rgba(0, 0, 0, 0.4);
    }
    
    /* Hero Header */
    .hero-header {
        background: linear-gradient(135deg, #1e293b, #0f172a);
        border-radius: 20px;
        padding: 30px;
        margin-bottom: 30px;
        border: 1px solid rgba(74, 144, 226, 0.2);
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
        position: relative;
        overflow: hidden;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .hero-header::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0; bottom: 0;
        background: radial-gradient(circle at top right, rgba(74, 144, 226, 0.15), transparent 50%);
        pointer-events: none;
    }
    .hero-title {
        margin: 0;
        font-size: 2rem;
        font-weight: 700;
        color: #ffffff;
    }
    .hero-subtitle {
        margin: 8px 0 0 0;
        color: #94a3b8;
        font-size: 1rem;
    }
    
    /* Custom Streamlit Buttons */
    .stButton > button {
        border-radius: 12px !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
    }
    .stButton > button[kind="secondary"] {
        background: rgba(255, 255, 255, 0.05) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        color: #e2e8f0 !important;
    }
    .stButton > button[kind="secondary"]:hover {
        background: rgba(74, 144, 226, 0.15) !important;
        border-color: rgba(74, 144, 226, 0.3) !important;
        transform: translateY(-2px) !important;
        color: #fff !important;
    }
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #4A90D9, #2C6BB3) !important;
        color: white !important;
        border: none !important;
        box-shadow: 0 4px 15px rgba(74, 144, 217, 0.3) !important;
    }
    .stButton > button[kind="primary"]:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(74, 144, 217, 0.5) !important;
        background: linear-gradient(135deg, #5CA0E9, #3A7BC3) !important;
    }
    
    /* Badges */
    .badge {
        display: inline-block;
        padding: 6px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        margin: 4px;
        backdrop-filter: blur(4px);
    }
    .badge-blue { background: rgba(74,144,217,0.15); color: #4A90D9; border: 1px solid rgba(74,144,217,0.3); }
    .badge-green { background: rgba(46,204,113,0.15); color: #2ecc71; border: 1px solid rgba(46,204,113,0.3); }
    .badge-orange { background: rgba(243,156,18,0.15); color: #f39c12; border: 1px solid rgba(243,156,18,0.3); }
    .badge-red { background: rgba(231,76,60,0.15); color: #e74c3c; border: 1px solid rgba(231,76,60,0.3); }
    
    /* Disclaimer / Alerts */
    .medical-alert {
        background: rgba(243, 156, 18, 0.1);
        border-left: 4px solid #f39c12;
        padding: 16px;
        border-radius: 12px;
        color: #f39c12;
        font-size: 0.95rem;
        margin: 20px 0;
        display: flex;
        align-items: center;
        gap: 12px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    /* Pipeline Step Card */
    .pipeline-card {
        text-align: center;
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }
    .pipeline-icon {
        font-size: 2.5rem;
        margin-bottom: 12px;
    }
    .pipeline-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #e2e8f0;
        margin-bottom: 16px;
    }
    
    /* Metric Cards */
    .small-metric {
        display: flex;
        align-items: center;
        gap: 15px;
    }
    .small-metric .icon-box {
        width: 48px;
        height: 48px;
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.5rem;
        background: rgba(255,255,255,0.05);
    }
    .small-metric-content h4 {
        margin: 0;
        font-size: 0.85rem;
        color: #94a3b8;
        font-weight: 500;
    }
    .small-metric-content h2 {
        margin: 4px 0 0 0;
        font-size: 1.4rem;
        color: #fff;
        font-weight: 700;
    }
    
    /* Risk Result Card */
    .risk-result {
        border-radius: 16px;
        padding: 30px;
        text-align: center;
        margin-bottom: 20px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }
    .risk-result.risk-0 { background: linear-gradient(135deg, rgba(46,204,113,0.2), rgba(46,204,113,0.05)); border: 2px solid rgba(46,204,113,0.5); }
    .risk-result.risk-1 { background: linear-gradient(135deg, rgba(243,156,18,0.2), rgba(243,156,18,0.05)); border: 2px solid rgba(243,156,18,0.5); }
    .risk-result.risk-2 { background: linear-gradient(135deg, rgba(231,76,60,0.2), rgba(231,76,60,0.05)); border: 2px solid rgba(231,76,60,0.5); }
    
    .risk-result h1 {
        margin: 0 0 10px 0;
        font-size: 2.5rem;
        font-weight: 800;
    }
    .risk-0 h1 { color: #2ecc71; }
    .risk-1 h1 { color: #f39c12; }
    .risk-2 h1 { color: #e74c3c; }
    </style>
    """, unsafe_allow_html=True)

def render_sidebar():
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "🏠 Genel Bakış"
        
    st.sidebar.markdown("""
        <div style="text-align:center; padding:20px 0;">
            <img src="https://img.icons8.com/fluency/96/heart-with-pulse.png" width="80" style="margin-bottom:10px;">
            <h2 style="margin:0; font-weight:700; background: linear-gradient(90deg, #4A90D9, #2ecc71); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">HibritML</h2>
            <p style="color:#94a3b8; font-size:0.85rem; margin-top:5px;">Kardiyovasküler Risk Analizi</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.sidebar.markdown("<br>", unsafe_allow_html=True)
    
    pages = [
        "🏠 Genel Bakış",
        "📂 Veri Yükle & Pipeline",
        "📊 Model Sonuçları",
        "🔬 Risk Tahmini"
    ]
    
    for title in pages:
        is_active = st.session_state.current_page == title
        btn_type = "primary" if is_active else "secondary"
        if st.sidebar.button(title, use_container_width=True, type=btn_type):
            st.session_state.current_page = title
            st.rerun()

    st.sidebar.markdown("---")
    st.sidebar.caption("Kaggle Cardiovascular Disease Dataset desteklenir.")
    
    return st.session_state.current_page

def render_hero_header():
    st.markdown("""
    <div class="hero-header">
        <div>
            <h1 class="hero-title">🫀 HibritML</h1>
            <p class="hero-subtitle">Bulanık Üç Değerli Mantık ve ML Algoritmaları ile Karşılaştırmalı Risk Tahmini</p>
        </div>
        <div style="text-align: right; display: flex; flex-direction: column; gap: 8px;">
            <div>
                <span class="badge badge-blue">SQLite Aktif</span>
                <span class="badge badge-green">Fuzzy Logic</span>
            </div>
            <div>
                <span class="badge badge-orange">7 ML Model</span>
                <span class="badge badge-blue">Streamlit Dashboard</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def render_metric_card(icon, title, value, color_class="blue"):
    st.markdown(f"""
    <div class="modern-card">
        <div class="small-metric">
            <div class="icon-box" style="color: var(--{color_class}); border: 1px solid rgba(255,255,255,0.1);">
                {icon}
            </div>
            <div class="small-metric-content">
                <h4>{title}</h4>
                <h2>{value}</h2>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def render_pipeline_step(icon, title, status, badge_color):
    st.markdown(f"""
    <div class="modern-card pipeline-card">
        <div class="pipeline-icon">{icon}</div>
        <div class="pipeline-title">{title}</div>
        <div style="margin-bottom: 15px;">
            <span class="badge badge-{badge_color}">{status}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

def render_risk_result_card(label, target_value, other_factors, rule_desc):
    # Mapping target_value to style classes
    style_class = f"risk-{target_value}"
    
    st.markdown(f"""
    <div class="risk-result {style_class}">
        <h1>{label}</h1>
        <div style="display: flex; justify-content: center; gap: 15px; margin-top: 20px;">
            <span class="badge badge-blue">OtherFactors: {other_factors}</span>
            <span class="badge badge-orange">Fuzzy Hedef: {target_value}</span>
        </div>
        <p style="margin-top: 20px; color: #cbd5e1; font-size: 0.95rem;">
            <b>Uygulanan Kural:</b><br>{rule_desc}
        </p>
    </div>
    """, unsafe_allow_html=True)

def render_footer_warning():
    st.markdown("""
    <div class="medical-alert">
        <span style="font-size: 1.5rem;">⚠️</span>
        <div>
            <b>Bu sistem eğitim ve araştırma amaçlıdır.</b> Tıbbi teşhis veya tedavi yerine geçmez. 
            Sağlık durumunuzla ilgili her türlü karar için lütfen uzman bir hekime danışınız.
        </div>
    </div>
    """, unsafe_allow_html=True)
