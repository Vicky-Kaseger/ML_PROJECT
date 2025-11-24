import pandas as pd
import numpy as np

# ==============================================================================
# KONFIGURASI FITUR (MENGIKUTI PERMINTAAN FILE MODEL .PKL YANG AKTIF)
# ==============================================================================

# Berdasarkan Error Traceback: Model Suhu SUDAH DILATIH dengan fitur ini:
# ['Suhu', 'Kelembapan', 'jam_dalam_hari', 'suhu_1jam_lalu', 'suhu_24jam_lalu', 'kelembapan_1jam_lalu']
FEATURES_SUHU = [
    'Suhu',
    'Kelembapan',
    'jam_dalam_hari',
    'suhu_1jam_lalu',
    'suhu_24jam_lalu',
    'kelembapan_1jam_lalu' 
]

# Fitur Hujan (Kita pertahankan sesuai info terakhir Anda)
FEATURES_HUJAN = [
    'CurahHujan',
    'jam_dalam_hari',
    'Kelembapan',
    'suhu_2jam_lalu',
    'hari_dalam_minggu'
]

# ==============================================================================
# FUNGSI UTILITAS
# ==============================================================================

def ensure_timezone(df, time_col='time'):
    """Memastikan kolom waktu memiliki timezone yang benar."""
    df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
    df = df.dropna(subset=[time_col])
    
    if df[time_col].dt.tz is None:
        df[time_col] = df[time_col].dt.tz_localize("UTC")
    
    # Convert ke WITA (Asia/Makassar)
    df[time_col] = df[time_col].dt.tz_convert("Asia/Makassar")
    return df

def normalize_columns(df):
    """
    Standarisasi nama kolom.
    """
    rename_map = {
        "Waktu": "time", 
        "temperature_2m (°C)": "Suhu",
        "relative_humidity_2m (%)": "Kelembapan",
        "rain (mm)": "CurahHujan",
        "weather_code (wmo code)": "DeskripsiCuaca",
    }
    df.rename(columns=rename_map, inplace=True)
    return df

def add_calendar_features(df, time_col='time'):
    df["jam_dalam_hari"] = df[time_col].dt.hour
    df["hari_dalam_minggu"] = df[time_col].dt.dayofweek
    return df

def add_lag_features(df):
    """
    Membuat fitur lag sesuai permintaan Model .PKL yang aktif.
    """
    
    # Helper aman untuk mengambil kolom
    def get_series(df, col_name):
        if col_name in df.columns:
            obj = df[col_name]
            if isinstance(obj, pd.DataFrame):
                return obj.iloc[:, 0]
            return obj
        return None

    s_suhu = get_series(df, "Suhu")
    s_hum = get_series(df, "Kelembapan")
    s_rain = get_series(df, "CurahHujan")

    # --- Fitur Lag Suhu ---
    # Model meminta: suhu_1jam_lalu, suhu_24jam_lalu, suhu_2jam_lalu
    if s_suhu is not None:
        df["suhu_1jam_lalu"] = s_suhu.shift(1)
        df["suhu_2jam_lalu"] = s_suhu.shift(2)
        df["suhu_24jam_lalu"] = s_suhu.shift(24)

    # --- Fitur Lag Kelembapan ---
    # ERROR SEBELUMNYA BILANG MODEL MINTA: 'kelembapan_1jam_lalu'
    if s_hum is not None:
        df["kelembapan_1jam_lalu"] = s_hum.shift(1)  # <-- KITA KEMBALIKAN KE 1 JAM
        
    # --- Fitur Lag Hujan ---
    if s_rain is not None:
        df["CurahHujan_24jam_lalu"] = s_rain.shift(24)
        
    return df

def prepare_input(df):
    """Pipeline preprocessing."""
    df_processed = df.copy()
    
    # Cek & Fix nama kolom 'time'
    if 'time' not in df_processed.columns:
        if 'Waktu' in df_processed.columns:
            df_processed.rename(columns={'Waktu': 'time'}, inplace=True)
        elif isinstance(df_processed.index, pd.DatetimeIndex):
            df_processed.reset_index(inplace=True)
            df_processed.rename(columns={'index': 'time'}, inplace=True)
        else:
            return pd.DataFrame()
    
    # 1. Standarisasi
    df_processed = ensure_timezone(df_processed, 'time')
    df_processed = normalize_columns(df_processed)
    
    # Hapus duplikat kolom nama sama
    df_processed = df_processed.loc[:, ~df_processed.columns.duplicated()]

    # 2. Urutkan Waktu & Handle Duplikat Data
    df_processed = df_processed.sort_values(by='time')
    df_processed = df_processed.drop_duplicates(subset=['time'], keep='last')

    # 3. Set Index & Resample
    df_processed.set_index('time', inplace=True)
    df_processed = df_processed.sort_index()

    df_resampled = df_processed.resample('H').ffill()
    df_resampled.reset_index(inplace=True)

    # 4. Feature Engineering
    df_final = add_calendar_features(df_resampled, 'time')
    
    # Safety check sebelum lag
    df_final = df_final.loc[:, ~df_final.columns.duplicated()]
    
    df_final = add_lag_features(df_final)
    
    # 5. Hapus NaN
    df_final.dropna(inplace=True)
    df_final.reset_index(drop=True, inplace=True)

    if df_final.empty:
        return pd.DataFrame()

    # Ambil baris terakhir
    last_row = df_final.iloc[[-1]].copy()

    # --- VALIDASI PENTING ---
    # Model Suhu MEMBUTUHKAN kolom 'Suhu' sebagai input fitur, bukan cuma lag.
    # Kita pastikan kolom 'Suhu' asli terbawa.
    if 'Suhu' not in last_row.columns and 'Suhu' in df_final.columns:
         last_row['Suhu'] = df_final['Suhu'].iloc[-1]

    return last_row