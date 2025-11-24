import gspread
import pandas as pd
import numpy as np
from google.oauth2.service_account import Credentials

SCOPES = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]

def get_client(json_path):
    creds = Credentials.from_service_account_file(json_path, scopes=SCOPES)
    client = gspread.authorize(creds)
    return client

def read_sheet(json_path, spreadsheet_id, sheet_name):
    """
    Membaca Google Sheet dari n8n dan menormalisasi header serta format angka.
    """
    client = get_client(json_path)
    sh = client.open_by_key(spreadsheet_id)
    ws = sh.worksheet(sheet_name)

    rows = ws.get_all_values()

    if not rows or len(rows) < 2:
        return pd.DataFrame()

    # Ambil header dan data
    header = [h.strip() for h in rows[0]]
    data_rows = rows[1:]

    df = pd.DataFrame(data_rows, columns=header)
    
    # --- 1. MAPPING HEADER DARI N8N (INDO) KE INTERAL SYSTEM ---
    # Kiri: Header di Google Sheet (dari n8n)
    # Kanan: Nama kolom yang dipakai di preprocessing.py
    rename_map = {
        "Waktu": "time",
        "Suhu": "Suhu",
        "Kelembapan": "Kelembapan",
        "CurahHujan": "CurahHujan",
        "DeskripsiCuaca": "DeskripsiCuaca",
        # Jaga-jaga jika n8n mengirim variasi lain
        "Temp": "Suhu",
        "RH": "Kelembapan",
        "Rain": "CurahHujan"
    }
    df.rename(columns=rename_map, inplace=True)
    
    # Hapus kolom kosong
    df = df.loc[:, df.columns != '']
    df.dropna(how='all', inplace=True)
    
    # --- 2. BERSIHKAN KOLOM WAKTU ---
    if 'time' in df.columns:
        df['time'].replace('', pd.NA, inplace=True)
        df.dropna(subset=['time'], inplace=True)
        # Pastikan format datetime bisa dibaca
        df['time'] = pd.to_datetime(df['time'], dayfirst=True, errors='coerce')
        # Hapus yang gagal jadi tanggal (NaT)
        df.dropna(subset=['time'], inplace=True)
    
    df = df.reset_index(drop=True)
    
    # --- 3. KONVERSI ANGKA (KOMA JADI TITIK) ---
    numeric_cols = ['Suhu', 'Kelembapan', 'CurahHujan']
    
    for col in numeric_cols:
        if col in df.columns:
            # Ganti koma dengan titik, lalu ubah ke float
            df[col] = df[col].astype(str).str.replace(',', '.', regex=False)
            df[col] = pd.to_numeric(df[col], errors='coerce')

    return df

def append_row(json_path, spreadsheet_id, sheet_name, row_values):
    client = get_client(json_path)
    sh = client.open_by_key(spreadsheet_id)
    ws = sh.worksheet(sheet_name)
    ws.append_row(row_values, value_input_option="USER_ENTERED", insert_data_option="INSERT_ROWS")