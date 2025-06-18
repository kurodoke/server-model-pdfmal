import os
import joblib
import pandas as pd
from flask import Flask, request, jsonify
from feature_extractor import extract_features
import xgboost as xgb
import re

# Inisialisasi aplikasi Flask
app = Flask(__name__)

# Path ke folder untuk menyimpan file upload sementara
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Muat model dan scaler saat aplikasi dimulai
try:
    # Memuat model dari format .json yang lebih stabil
    model = xgb.Booster()
    model.load_model("model.json")
    
    # Memuat scaler
    scaler = joblib.load("scaler.pkl")
    print("Model dan scaler berhasil dimuat.")
except FileNotFoundError as e:
    print(f"Error: {e}. Pastikan file 'model.json' dan 'scaler.pkl' ada.")
    model = None
    scaler = None

# Fungsi untuk mengekstrak informasi header PDF
def header_obj(col):
  match = bool(re.fullmatch(r'\s*%PDF-\d\.\d\s*', col))
  if match:
    version = col.split('%PDF-')[1][:5]  # ambil 5 karakter pertama setelah '%PDF-'
    return float(version)
  return -1

# Definisikan endpoint untuk prediksi
@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint untuk menerima PDF dan mengembalikan prediksi."""
    if model is None or scaler is None:
        return jsonify({"error": "Model atau scaler tidak dapat dimuat. Periksa log server."}), 500

    if 'file' not in request.files:
        return jsonify({"error": "Tidak ada file yang dikirim"}), 400
    
    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "File tidak dipilih"}), 400

    if file and file.filename.lower().endswith('.pdf'):
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        
        print(f"File diterima: {file.filename}")

        # 1. Ekstraksi Fitur
        try:
            df_features = extract_features(filepath)
            if df_features is None:
                raise ValueError("Ekstraksi fitur gagal.")
            print("Fitur berhasil diekstraksi.")
        except Exception as e:
            os.remove(filepath)
            return jsonify({"error": f"Gagal mengekstrak fitur: {e}"}), 500

        # 2. Preprocessing
        df_processed = df_features.copy()
        filename = df_processed["FileName"].iloc[0]

        features_to_drop = [
            'JS', 'TitleCharacters', 'Endstream', 'Endobj', 'Trailer', 'FileName'
        ]
        
        for col in features_to_drop:
            if col in df_processed.columns:
                df_processed.drop(columns=[col], inplace=True)
            else:
                 print(f"Peringatan: Kolom '{col}' tidak ditemukan untuk dihapus.")

        df_processed['Header'] = df_processed['Header'].apply(lambda col: header_obj(col))

        print("Pra-pemrosesan selesai.")
        
        # 3. Prediksi Model
        try:
            # a. Skalakan data seperti sebelumnya
            scaled_features = scaler.transform(df_processed)
            
            # b. Ubah data ke format DMatrix yang dikenali XGBoost
            dmatrix_features = xgb.DMatrix(scaled_features)

            # c. Lakukan prediksi. Ini akan menghasilkan probabilitas untuk kelas '1' (Malicious)
            prob_malicious = model.predict(dmatrix_features)[0]

            # d. Tentukan label dan probabilitas final berdasarkan ambang batas (threshold 0.5)
            if prob_malicious > 0.5:
                prediction_label = "Malicious"
                probability_for_display = float(prob_malicious) # Konversi ke float standar
            else:
                prediction_label = "Benign"
                probability_for_display = 1.0 - float(prob_malicious) # Konversi ke float standar
            
            print(f"Prediksi: {prediction_label}, Probabilitas: {probability_for_display:.2%}")

        except Exception as e:
            os.remove(filepath)
            return jsonify({"error": f"Gagal saat prediksi: {e}"}), 500

        # Hapus file sementara setelah selesai
        os.remove(filepath)

        # 4. Kirim Respons
        return jsonify({
            "filename": filename,
            "prediction": prediction_label,
            "probability": f"{probability_for_display:.2%}"
        })

    else:
        return jsonify({"error": "Format file tidak didukung. Harap unggah file PDF."}), 400

# Jalankan server
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)