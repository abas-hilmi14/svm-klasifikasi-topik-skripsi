import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Judul aplikasi
st.set_page_config(page_title="Klasifikasi Topik Skripsi", layout="centered")
st.title("🚀 Klasifikasi Topik Skripsi Mahasiswa")
st.markdown("Model klasifikasi berbasis **SVM + ADASYN + Z-Score + Information Gain**")

# Load model dan komponen
scaler = joblib.load("scaler.pkl")
model = joblib.load("model_svm.pkl")
label_encoder = joblib.load("label_encoder.pkl")
selected_features = joblib.load("selected_features.pkl")

# Input manual
st.subheader("🔢 Masukkan Nilai Mata Kuliah")
user_input = {}
for feature in selected_features:
    user_input[feature] = st.number_input(f"{feature}", min_value=0.0, max_value=100.0,value=80, step=0.1)

# Jika tombol prediksi ditekan
if st.button("🔍 Prediksi Topik Skripsi"):
    # Ambil inputan dan bentuk ke array
    input_df = pd.DataFrame([user_input])
    
    # Normalisasi Z-Score
    input_scaled = scaler.transform(input_df)

    # Prediksi kelas
    pred = model.predict(input_scaled)
    pred_label = label_encoder.inverse_transform(pred)[0]

    # Tampilkan hasil
    st.success(f"📌 **Topik Skripsi Diprediksi: {pred_label}**")
