# Proyek Sains Data: Klasifikasi & Deteksi Aliran Air Tempat Wudhu Masjid

Sistem cerdas berbasis Machine Learning (*Ensemble ExtraTrees + Tuned Random Forest*) dengan antarmuka interaktif **Streamlit** untuk mendeteksi dan mengklasifikasikan aliran air kran di tempat wudhu masjid secara akustik real-time, guna mencegah pemborosan air dan mendukung konservasi air masjid.

Proyek ini telah dikontainerisasi menggunakan **Docker** dan siap dideploy langsung pada **Server Lab Proxmox** hanya dengan satu perintah.

---

## 3 Kelas Klasifikasi

| Label | Kelas | Karakteristik Akustik | Tindakan Sistem |
|:---:|---|---|---|
| **0** | **Bukan Air** | Suara kipas angin, obrolan, ketikan keyboard, kumandang adzan, kebisingan lingkungan | Tidak memicu alarm sistem (kran tertutup) |
| **1** | **Kran Pelan (Hemat)** | Aliran air wajar, laminar, hemat air wudhu | Peringatan jika mengalir terus-menerus **> 20 detik** (indikasi lupa mematikan kran) |
| **2** | **Kran Keras (Boros)** | Aliran air bertekanan tinggi, turbulen, cipratan deras | Peringatan jika mengalir terus-menerus **> 10 detik** (indikasi pemborosan air wudhu) |

---

## Arsitektur & Performa Model

- **Jumlah Fitur Akustik**: 81 fitur (Spectral Centroid, Bandwidth, Rolloff, ZCR, HPSS Harmonic Ratio, Spectral Crest & Flux, Sub-band Energy Ratios, 20 MFCC + Delta, dan Spectral Contrast).
- **Algoritma**: Soft-Voting Ensemble (`ExtraTreesClassifier` + `RandomForestClassifier`).
- **Akurasi Data Uji (Test Accuracy)**: **97.31%**
- **5-Fold Stratified Cross-Validation**: **97.54% (±0.24%)**

---

## Cara Menjalankan Menggunakan Docker (Server Lab Proxmox)

> [!IMPORTANT]
> Sesuai ketentuan, aplikasi dapat langsung dijalankan hanya dengan satu perintah:

```bash
docker compose up --build
```

Setelah proses build selesai, buka browser dan akses aplikasi melalui:
```text
http://localhost:8501
```
atau jika diakses melalui IP Server Lab:
```text
http://<IP-SERVER-LAB>:8501
```

Port yang dibuka: **8501** (hanya 1 port).

---

## Fitur Aplikasi Antarmuka Streamlit

### 1. Tab Deteksi & Klasifikasi Suara
- **Upload File Audio**: Mendukung format `.wav`, `.mp3`, `.m4a`, `.ogg`, dan `.flac`.
- **Rekam Mikrofon Langsung**: Menggunakan widget perekam suara langsung dari mikrofon peramban client.
- **Tombol Sampel Bawaan (Quick Demo)**: Uji coba instan dengan satu klik menggunakan sampel *Kran Keras*, *Kran Pelan*, dan *Bukan Air* yang telah disediakan.
- **Peringatan Konservasi Air**: Banner rekomendasi tindakan otomatis (peringatan debit kran deras vs kran ramah lingkungan).
- **Distribusi Probabilitas Softmax**: Visualisasi confidence level ketiga kelas.
- **Deteksi Per Detik (Timeline Graph)**: Grafik prediksi dinamis per interval 2 detik sepanjang durasi audio.

### 2. Tab Visualisasi Akustik & Spektrogram
- **Waveform Plot**: Visualisasi sinyal amplitudo audio terhadap waktu.
- **Mel-Spectrogram**: Pola persebaran energi frekuensi audio (0–8000 Hz) menggunakan color map visual.
- **Metrik Akustik**: RMS Energy, Spectral Centroid (Hz), dan Zero Crossing Rate.

### 3. Tab Info Model & Metodologi
- Dokumentasi 81 ekstraksi fitur akustik dan arsitektur model Ensemble.

---

## Cara Menjalankan Lokal (Tanpa Docker)

Jika ingin menjalankan aplikasi langsung di environment Python lokal:

1. **Install dependensi**:
   ```powershell
   pip install -r requirements.txt
   ```
2. **Jalankan Aplikasi Streamlit**:
   ```powershell
   streamlit run app.py
   ```
   Aplikasi akan otomatis terbuka di browser pada `http://localhost:8501`.

3. **Menjalankan Monitoring Mikrofon Laptop Real-Time (dengan Voice Warning)**:
   ```powershell
   python live_mic.py
   ```

4. **Melatih Ulang Model**:
   ```powershell
   python train.py
   ```

---

## Informasi Model & Dataset

- **Model Terlatih**: File model ensemble biner (`output/model_aliran_air.pkl`, ~47 MB) **sudah disertakan langsung di repository**, sehingga aplikasi dapat langsung digunakan tanpa perlu melakukan training ulang terlebih dahulu.
- **Sampel Audio Uji**: Disediakan 3 sampel representatif di folder `samples/` (`sample_kran_keras.mp3`, `sample_kran_pelan.mp3`, dan `sample_bukan_air.wav`).
- **Dataset Training**: Dataset audio mentah disimpan di folder `dataset/` (terdiri dari kategori `bukan_air`, `pelan`, dan `keras`). Untuk meregenerasi seluruh variasi data augmentasi, jalankan:
  ```powershell
  python augment_dataset.py
  ```

---

## Struktur Repositori

```text
proyek_hemat_air/
├── Dockerfile                  <-- Instruksi build image container Docker (Streamlit)
├── docker-compose.yml          <-- Konfigurasi docker compose untuk Proxmox (Port 8501)
├── requirements.txt            <-- Daftar dependensi Python
├── app.py                      <-- Aplikasi Antarmuka Utama (Streamlit)
├── flask_app.py                <-- Alternatif REST API & Web Dashboard (Flask)
├── samples/                    <-- File sampel audio bawaan untuk pengujian cepat
│   ├── sample_kran_keras.mp3
│   ├── sample_kran_pelan.mp3
│   └── sample_bukan_air.wav
├── output/
│   ├── model_aliran_air.pkl    <-- Model terlatih Ensemble ExtraTrees + RF (47 MB)
│   └── evaluasi_model.png      <-- Visualisasi Confusion Matrix & Top Features
├── dataset/                    <-- Audio dataset (bukan_air, pelan, keras)
├── train.py                    <-- Script pelatihan & ekstraksi 81 fitur audio
├── live_mic.py                 <-- Script monitoring real-time via mic laptop
├── predict.py                  <-- Script CLI klasifikasi satu file
├── augment_dataset.py          <-- Generator augmentasi akustik
├── README.md                   <-- Dokumentasi lengkap proyek
├── .dockerignore               <-- Filter build Docker
└── .gitignore                  <-- Filter commit Git
```

---

## Identitas Kelompok

- **Mata Kuliah**: Proyek Sains Data (PSD)
- **Dosen Pengampu**: Pak Mul
- **Topik**: Klasifikasi Audio / Akustik Aliran Air Tempat Wudhu Masjid
- **Anggota Kelompok**:
1. Raihan Aryanova Narendra - 240411100094
2. Muhammad Farhan - 240411100233
3. Joni Kumar Meghwar - 240411100234
