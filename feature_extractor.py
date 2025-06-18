import os
import subprocess
import pandas as pd
import fitz  # PyMuPDF
import tempfile

def extract_features(pdf_path):
    """
    Mengekstrak fitur dari satu file PDF.

    Args:
        pdf_path (str): Path ke file PDF.

    Returns:
        pandas.DataFrame: DataFrame yang berisi fitur-fitur yang diekstraksi,
                          atau None jika terjadi error.
    """
    try:
        # =================================================================
        # Bagian 1: Ekstraksi Fitur Umum menggunakan Fitz (PyMuPDF)
        # =================================================================
        doc = fitz.open(pdf_path)
        
        # Metadata dan Title
        metadata = doc.metadata
        title = metadata.get("title", "") if metadata else ""
        
        # Fitur-fitur dasar
        is_encrypted = 1 if doc.is_encrypted else 0
        num_pages = doc.page_count
        xref_length = doc.xref_length()
        pdf_size = int(os.path.getsize(pdf_path) / 1000)
        metadata_size = len(str(metadata).encode('utf-8'))
        title_chars = len(title) if title else 0
        
        # Ekstraksi teks untuk pengecekan
        text_found = 0
        text = ""
        try:
            for page in doc:
                text += page.get_text()
                if len(text) > 100:
                    text_found = 1
                    break
        except Exception:
            text_found = -1

        # Jumlah gambar dan file tersemat
        embed_count = doc.embfile_count()
        img_count = 0
        for k in range(len(doc)):
            try:
                img_count = len(doc.get_page_images(k)) + img_count
            except:
                img_count = -1
                break

        # Simpan fitur umum ke dalam dictionary
        general_features = {
            "FileName": os.path.basename(pdf_path),
            "PdfSize": pdf_size,
            "MetadataSize": metadata_size,
            "Pages": num_pages,
            "XrefLength": xref_length,
            "TitleCharacters": title_chars,
            "isEncrypted": is_encrypted,
            "EmbeddedFiles": embed_count,
            "Text": text_found,
        }
        
        df_general = pd.DataFrame([general_features])

        # =================================================================
        # Bagian 2: Ekstraksi Fitur Struktural menggunakan pdfid.py
        # =================================================================
        pdfid_dir = os.path.join(os.path.dirname(__file__), 'pdfid')
        pdfid_script = os.path.join(pdfid_dir, 'pdfid.py')
        
        # Pastikan path pdfid.py benar
        if not os.path.exists(pdfid_script):
            raise FileNotFoundError(f"pdfid.py tidak ditemukan di {pdfid_script}")

        # Perintah untuk menjalankan pdfid.py dan memformat outputnya
        cmd = [
            "python", pdfid_script, os.path.abspath(pdf_path)
        ]
        
        # Menjalankan pdfid dan mengambil output
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=pdfid_dir)
        
        if result.returncode != 0:
            print(f"Error menjalankan pdfid: {result.stderr}")
            return None

        # Parsing output dari pdfid
        pdfid_features = {}
        lines = result.stdout.strip().split('\n')
        
        # Header untuk fitur pdfid
        header = [
            "Header", "Obj", "Endobj", "Stream", "Endstream", "Xref", "Trailer", "StartXref",
            "PageNo", "Encrypt", "ObjStm", "JS", "Javascript", "AA", "OpenAction",
            "Acroform", "JBIG2Decode", "RichMedia", "Launch", "EmbeddedFile", "XFA",
            "URI", "Colors"
        ]
        
        # Inisialisasi semua fitur dengan 0
        for h in header:
            pdfid_features[h] = 0

        index = 0
        for line in lines:
            if 'PDFiD' in line:
                continue
            parts = line.strip().split()
            if len(parts) >= 2:
                feature_value = parts[1]
                
                if parts[0] == '/Colors':
                    feature_value = parts[3]    
                
                pdfid_features[header[index]] = feature_value
                index += 1

        df_pdfid = pd.DataFrame([pdfid_features])

        # =================================================================
        # Bagian 3: Menggabungkan Semua Fitur
        # =================================================================
        final_df = pd.concat([df_general, df_pdfid], axis=1)
        
        # Memastikan kolom 'Text' adalah kategori untuk konsistensi

        return final_df

    except Exception as e:
        print(f"Terjadi error saat mengekstrak fitur: {e}")
        return None