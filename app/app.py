# ==========================================
# UNSRAT WEATHER ASSISTANT - STREAMLIT APP
# ==========================================
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime

# --- IMPORT MODULES DARI FOLDER UTILS ---
try:
    from utils.google_sheets import read_sheet, append_row
    from utils.preprocessing import prepare_input, FEATURES_SUHU, FEATURES_HUJAN
except ImportError as e:
    st.error(f"Gagal mengimport modul dari folder 'utils'. Pastikan file ada. Error: {e}")
    st.stop()

# ==========================================
# KONFIGURASI APLIKASI
# ==========================================
st.set_page_config(
    page_title="UNSRAT Climate AI",
    page_icon="🌌",
    layout="centered"
)
# --- [BARU] FITUR GANTI TEMA BACKGROUND ---
# --- [UPDATE] FITUR GANTI TEMA BACKGROUND (MODE MALAM) ---
def inject_custom_css():
    with st.sidebar:
        # st.markdown("### 🎨 Tampilan")
        # Toggle Button dengan label baru
        tema_malam = st.toggle("", value=False)

    if tema_malam:
        # CSS untuk Tema Malam (Dark Blue Gradient)
        st.markdown(
            """
            <style>
            /* 1. Background Utama: Gradasi Biru Tua Malam (Midnight) */
            .stApp {
                background: linear-gradient(to bottom, #0f0c29, #302b63, #24243e);
                background-attachment: fixed;
                color: white; /* Mengubah teks dasar jadi putih */
            }
            
            /* 2. Mengubah Sidebar menjadi Gelap Transparan */
            [data-testid="stSidebar"] {
                background-color: rgba(15, 12, 41, 0.95);
                border-right: 1px solid #4b6cb7;
            }
            
            /* 3. Memaksa Judul dan Teks agar berwarna Putih */
            h1, h2, h3, h4, h5, h6, p, span, label {
                color: #ffffff !important;
            }
            
            /* 4. Membuat Angka Metric (Suhu) berwarna Cyan Terang (Glowing) */
            [data-testid="stMetricValue"] {
                color: #00d2ff !important;
                font-weight: bold;
                text-shadow: 0 0 10px rgba(0, 210, 255, 0.5); /* Efek glowing */
            }
            
            /* 5. Label Metric (Tulisan 'Suhu', 'Hujan') jadi abu-abu terang */
            [data-testid="stMetricLabel"] {
                color: #e0e0e0 !important;
            }

            /* 6. Menyesuaikan warna notifikasi (Success/Error/Warning) agar tetap terbaca */
            .stAlert {
                background-color: rgba(255, 255, 255, 0.1);
                color: white;
                border: 1px solid white;
            }
            </style>
            """,
            unsafe_allow_html=True
        )
    else:
        # CSS Default (Minimalis / Putih)
        st.markdown(
            """
            <style>
            .stApp {
                background-color: #FFFFFF;
                color: black;
            }
            </style>
            """,
            unsafe_allow_html=True
        )

# Panggil fungsi ini agar CSS dijalankan
inject_custom_css()
# Konfigurasi Google Sheets
CREDENTIALS_PATH = "utils/beaming-ring-478707-m1-2dd3d047f00d.json" 
SPREADSHEET_ID = "1jivwowHS44dyIgpMTqQwnDdIYZTMaU3NIoEzhsnUWHs"
SHEET_NAME = "Sheet1"

# ==========================================
# FUNGSI LOAD MODEL
# ==========================================
@st.cache_resource
def load_models():
    """Memuat semua model yang telah dilatih dari folder 'models/'."""
    models = {}
    try:
        files = {
            "suhu_1h": "model/suhu/suhu_1h.pkl",
            "suhu_3h": "model/suhu/suhu_3h.pkl",
            "suhu_6h": "model/suhu/suhu_6h.pkl",
            "hujan_1h": "model/curahHujan/hujan_1h.pkl",
            "hujan_3h": "model/curahHujan/hujan_3h.pkl",
            "hujan_6h": "model/curahHujan/hujan_6h.pkl"
        }
        
        for key, path in files.items():
            if os.path.exists(path):
                models[key] = joblib.load(path)
            else:
                st.error(f"File model tidak ditemukan: {path}")
                return None
        return models
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memuat model: {e}")
        return None

# ==========================================
# FUNGSI REKOMENDASI (UNTUK KLASIFIKASI)
# ==========================================
# ==========================================
# FUNGSI REKOMENDASI (SCIENTIFIC UPGRADE)
# ==========================================
def calculate_heat_index(T, RH):
    """
    Menghitung Heat Index (Suhu Terasa) berdasarkan rumus Rothfusz (NWS).
    T dalam Celcius, RH dalam Persen.
    """
    # Rumus sederhana Heat Index (Steadman) untuk screening awal
    # HI = 0.5 * (T + 61.0 + ((T-68.0)*1.2) + (RH*0.094)) -> Ini versi Fahrenheit
    
    # Kita gunakan pendekatan Sederhana untuk Iklim Tropis (Celsius):
    # Heat Index ≈ T + 0.555 * (e - 10) 
    # Dimana e adalah tekanan uap air.
    
    # Tapi untuk mempermudah coding tanpa rumus uap air yang kompleks, 
    # kita gunakan logika threshold gabungan yang umum di jurnal kesehatan tropis:
    
    feels_like = T
    
    # Jika suhu > 27 dan lembap > 80% (Khas Manado), rasa panas naik drastis
    if T >= 27:
        if RH >= 85:
            feels_like = T + 4  # Terasa jauh lebih panas
        elif RH >= 70:
            feels_like = T + 2  # Terasa agak lebih panas
            
    return feels_like

def get_recommendation_classification(pred_temp, pred_rain_class, pred_humidity=None):
    """
    Menerjemahkan kelas hujan dan suhu menjadi saran aksi.
    Update: Menambahkan logika Heat Index sederhana.
    """
    
    # --- 1. LOGIKA HUJAN (Sesuai output Model Klasifikasi) ---
    if pred_rain_class == 0:
        status_hujan = "Cerah / Berawan"
        icon = "☁️"
        pred_mm_display = "0 mm/jam"
    elif pred_rain_class == 1:
        status_hujan = "Hujan Ringan - Sedang" # Diperjelas
        icon = "🌧️"
        pred_mm_display = "1 - 5 mm/jam" # Estimasi BMKG
    else: # pred_rain_class == 2
        status_hujan = "Hujan Lebat"
        icon = "⛈️"
        pred_mm_display = "> 5 mm/jam"
        
    # --- 2. LOGIKA SUHU (THERMAL COMFORT) ---
    # Default kelembapan 80% jika tidak dipassing (rata-rata Manado)
    rh_val = pred_humidity if pred_humidity is not None else 80 
    
    real_feel = calculate_heat_index(pred_temp, rh_val)
    
    status_suhu = "Nyaman"
    if real_feel < 25: 
        status_suhu = "Sejuk"
    elif 25 <= real_feel <= 29:
        status_suhu = "Normal"
    elif 29 < real_feel <= 33:
        status_suhu = "Hangat"
    else: # > 33
        status_suhu = "Panas Menyengat"
        
    # --- 3. REKOMENDASI AKSI ---
    saran = "✅ Cuaca kondusif untuk aktivitas kampus."
    warna = "success"
    
    # Prioritas 1: Hujan Ekstrem
    if pred_rain_class == 2:
        saran = "⚠️ **BAHAYA:** Potensi Hujan Lebat! Sebaiknya tunggu di dalam gedung/kantin. Hindari area parkir terbuka."
        warna = "error"
        
    # Prioritas 2: Hujan Ringan
    elif pred_rain_class == 1:
        saran = "☔ **Sedia Payung/Jas Hujan:** Akan turun hujan ringan. Lantai koridor mungkin licin."
        warna = "info"
        
    # Prioritas 3: Panas Ekstrem (Heat Stress)
    elif status_suhu == "Panas Menyengat":
        saran = "☀️ **Suhu Terasa Panas:** Hidrasi cukup dan gunakan pakaian yang menyerap keringat. Hindari aktivitas fisik berat di lapangan."
        warna = "warning"
            
    return status_hujan, status_suhu, icon, saran, warna, pred_mm_display
# ==========================================
# INTERFACE UTAMA (UI)
# ==========================================
# Judul Utama dengan sedikit styling agar lebih besar
st.markdown("<h1 style='text-align: center;'>🌌 UNSRAT Climate AI</h1>", unsafe_allow_html=True)

# Deskripsi / Sub-judul
st.markdown(
    """
    <p style='text-align: center; font-size: 1.2em; opacity: 0.8;'>
    Asisten cerdas pemantau cuaca mikro kampus berbasis <i>Machine Learning</i>.
    <br>
    Pantau suhu, hujan, dan kenyamanan termal secara <b>Real-time</b>.
    </p>
    """, 
    unsafe_allow_html=True
)

models_dict = load_models()

if models_dict:
    with st.sidebar:
        st.header("📡 Input Data Sensor")
        with st.form("input_form"):
            # --- INPUT DATA ---
            st.subheader("Kondisi Saat Ini")
            suhu_now = st.number_input("Suhu (°C)", 20.0, 40.0, 28.5, 0.1)
            kelembapan_now = st.number_input("Kelembapan (%)", 30, 100, 80)
            curah_now = st.number_input("Curah Hujan (mm)", 0.0, 100.0, 0.0, 0.1)
            
            waktu_skrg = datetime.now()
            jam_now = st.slider("Jam Saat Ini (WITA)", 0, 23, waktu_skrg.hour)
            
            st.divider()
            
            # --- DROPDOWN PILIHAN WAKTU PREDIKSI ---
            st.subheader("🎯 Target Prediksi")
            pilihan_waktu = st.selectbox(
                "Ingin melihat prediksi untuk:",
                ("1 Jam ke Depan", "3 Jam ke Depan", "6 Jam ke Depan")
            )
            
            # Mapping pilihan text ke angka jam
            map_jam = {
                "1 Jam ke Depan": 1,
                "3 Jam ke Depan": 3,
                "6 Jam ke Depan": 6
            }
            target_h = map_jam[pilihan_waktu]
            
            st.divider()
            submit_btn = st.form_submit_button("🔍 Analisis Cuaca", type="primary")

    if submit_btn:
        with st.spinner("Mengambil data historis & memproses prediksi..."):
            try:
                df_history = read_sheet(CREDENTIALS_PATH, SPREADSHEET_ID, SHEET_NAME)
                
                # Gunakan nama kolom yang sudah distandarisasi sistem (Suhu, Kelembapan, dst)
                current_row = {
                    'time': waktu_skrg.replace(hour=jam_now, minute=0, second=0).isoformat(),
                    'Suhu': suhu_now,
                    'Kelembapan': kelembapan_now,
                    'CurahHujan': curah_now,
                    'DeskripsiCuaca': 0 # Dummy
                }
                
                df_current = pd.DataFrame([current_row])
                
                if not df_history.empty:
                    df_combined = pd.concat([df_history, df_current], ignore_index=True)
                else:
                    df_combined = df_current
                
                # Preprocessing
                X_processed = prepare_input(df_combined)
                
                if X_processed.empty:
                    st.error("Gagal membuat fitur prediksi. Data historis tidak cukup/valid.")
                    st.stop()
                
                # Filter fitur sesuai model
                X_final_suhu = X_processed[FEATURES_SUHU]
                X_final_hujan = X_processed[FEATURES_HUJAN]
                
                # ==========================================
                # TAMPILAN HASIL (SINGLE VIEW)
                # ==========================================
                st.divider()
                st.subheader(f"🔮 Hasil Peramalan: {pilihan_waktu}")
                
                # Prediksi HANYA untuk jam yang dipilih (target_h)
                model_suhu_key = f"suhu_{target_h}h"
                model_hujan_key = f"hujan_{target_h}h"
                
                pred_t = models_dict[model_suhu_key].predict(X_final_suhu)[0]
                pred_r_class = models_dict[model_hujan_key].predict(X_final_hujan)[0]
                
                h_txt, t_txt, icon, saran, color, pred_mm_display = get_recommendation_classification(pred_t, pred_r_class)
                
                # Tampilan Card Besar
                col_res1, col_res2 = st.columns([1, 2])
                
                with col_res1:
                    st.markdown(f"<h1 style='text-align: center; font-size: 80px;'>{icon}</h1>", unsafe_allow_html=True)
                    st.markdown(f"<h3 style='text-align: center;'>{h_txt}</h3>", unsafe_allow_html=True)
                
                with col_res2:
                    st.markdown("### Detail Angka")
                    c1, c2 = st.columns(2)
                    c1.metric("🌡️ Prediksi Suhu", f"{pred_t:.1f}°C", delta=t_txt, delta_color="off")
                    c2.metric("💧 Intensitas Hujan", pred_mm_display)
                    
                    st.markdown("### 💡 Rekomendasi")
                    if color == "error":
                        st.error(saran)
                    elif color == "warning":
                        st.warning(saran)
                    elif color == "info":
                        st.info(saran)
                    else:
                        st.success(saran)
                
            except Exception as e:
                st.error("Terjadi kesalahan sistem saat prediksi:")
                st.exception(e)

    else:
        st.info("👈 Silahkan pilih target waktu prediksi (1, 3, atau 6 jam) di panel sebelah kiri dan klik Analisis.")

else:
    st.warning("Gagal memuat file model. Pastikan file model ada di folder 'models/'.")