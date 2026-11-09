"""
app.py
======
Aplikasi Streamlit - Sistem Cerdas Klasifikasi & Deteksi Aliran Air Tempat Wudhu Masjid
Berbasis Machine Learning (Ensemble ExtraTrees + Tuned Random Forest).

Dapat dijalankan langsung dengan:
    streamlit run app.py
atau melalui Docker:
    docker compose up --build
"""

import os
import io
import tempfile
import warnings
import joblib
import numpy as np
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt
import streamlit as st

warnings.filterwarnings('ignore')

# ==========================================
# KONFIGURASI HALAMAN STREAMLIT
# ==========================================
st.set_page_config(
    page_title="Klasifikasi Aliran Air Tempat Wudhu Masjid",
    page_icon="🚰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (CSS)
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #0284c7 0%, #0369a1 50%, #075985 100%);
        padding: 24px 28px;
        border-radius: 14px;
        color: white;
        margin-bottom: 24px;
        box-shadow: 0 10px 25px -5px rgba(2, 132, 199, 0.3);
    }
    .main-header h1 {
        font-size: 1.85rem;
        font-weight: 700;
        margin-bottom: 6px;
        color: #ffffff;
    }
    .main-header p {
        font-size: 0.98rem;
        margin-bottom: 0px;
        color: #e0f2fe;
    }
    .badge-psd {
        background-color: #38bdf8;
        color: #0c4a6e;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.5px;
        display: inline-block;
        margin-bottom: 8px;
    }
    .alert-card {
        padding: 20px;
        border-radius: 12px;
        margin-top: 15px;
        margin-bottom: 20px;
        font-weight: 500;
    }
    .alert-keras {
        background-color: #fef2f2;
        border: 2px solid #ef4444;
        color: #991b1b;
    }
    .alert-pelan {
        background-color: #f0fdf4;
        border: 2px solid #22c55e;
        color: #166534;
    }
    .alert-bukan {
        background-color: #f8fafc;
        border: 2px solid #94a3b8;
        color: #334155;
    }
    .metric-box {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 14px;
        text-align: center;
    }
    .metric-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: #0284c7;
    }
    .metric-label {
        font-size: 0.82rem;
        color: #64748b;
    }
</style>
""", unsafe_allow_html=True)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "output", "model_aliran_air.pkl")
SAMPLES_DIR = os.path.join(BASE_DIR, "samples")

CLASS_NAMES = [
    "Bukan Air (Kipas/Adzan/Suasana)",
    "Kran Pelan (Hemat)",
    "Kran Keras (Boros)"
]

# ==========================================
# LOAD MODEL & EXTRACT FEATURES
# ==========================================
@st.cache_resource
def load_ml_model():
    if not os.path.exists(MODEL_PATH):
        return None, None, CLASS_NAMES
    try:
        model_data = joblib.load(MODEL_PATH)
        model = model_data.get("model", model_data)
        feature_cols = model_data.get("feature_cols", None)
        class_names = model_data.get("class_names", CLASS_NAMES)
        return model, feature_cols, class_names
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        return None, None, CLASS_NAMES

def extract_segment_features(seg, sr=22050):
    """Ekstraksi 81 fitur akustik per segmen audio (2 detik)."""
    rms_frames = librosa.feature.rms(y=seg)
    rms_val = float(np.mean(rms_frames))
    spec_cent = float(np.mean(librosa.feature.spectral_centroid(y=seg, sr=sr)))
    spec_bw = float(np.mean(librosa.feature.spectral_bandwidth(y=seg, sr=sr)))
    spec_ro85 = float(np.mean(librosa.feature.spectral_rolloff(y=seg, sr=sr, roll_percent=0.85)))
    spec_ro25 = float(np.mean(librosa.feature.spectral_rolloff(y=seg, sr=sr, roll_percent=0.25)))
    spec_flat = float(np.mean(librosa.feature.spectral_flatness(y=seg)))
    zcr_frames = librosa.feature.zero_crossing_rate(y=seg)
    zcr_val = float(np.mean(zcr_frames))
    zcr_std = float(np.std(zcr_frames))

    y_harm, y_perc = librosa.effects.hpss(seg, margin=(2.0, 1.0))
    harm_rms = float(np.mean(librosa.feature.rms(y=y_harm)))
    harm_ratio = float(harm_rms / (rms_val + 1e-9))

    S = np.abs(librosa.stft(seg))
    crest = float(np.mean(np.max(S, axis=0) / (np.mean(S, axis=0) + 1e-9)))
    flux = float(np.mean(librosa.onset.onset_strength(y=seg, sr=sr)))

    freqs = librosa.fft_frequencies(sr=sr)
    low_idx = np.where(freqs < 250)[0]
    mid_idx = np.where((freqs >= 250) & (freqs < 3000))[0]
    high_idx = np.where(freqs >= 3000)[0]

    low_e = float(np.sum(S[low_idx, :]**2))
    mid_e = float(np.sum(S[mid_idx, :]**2))
    high_e = float(np.sum(S[high_idx, :]**2))
    tot_e = low_e + mid_e + high_e + 1e-9

    low_ratio = float(low_e / tot_e)
    mid_ratio = float(mid_e / tot_e)
    high_ratio = float(high_e / tot_e)

    mfcc = librosa.feature.mfcc(y=seg, sr=sr, n_mfcc=20)
    mfcc_delta = librosa.feature.delta(mfcc)
    contrast = librosa.feature.spectral_contrast(y=seg, sr=sr)

    row_dict = {
        'rms': rms_val,
        'spectral_centroid': spec_cent,
        'spectral_bandwidth': spec_bw,
        'spectral_rolloff': spec_ro85,
        'spectral_rolloff_25': spec_ro25,
        'spectral_flatness': spec_flat,
        'zcr': zcr_val,
        'zcr_std': zcr_std,
        'harm_ratio': harm_ratio,
        'crest': crest,
        'flux': flux,
        'low_ratio': low_ratio,
        'mid_ratio': mid_ratio,
        'high_ratio': high_ratio,
    }
    for i in range(20):
        row_dict[f'mfcc_mean_{i+1}'] = float(np.mean(mfcc[i]))
        row_dict[f'mfcc_std_{i+1}'] = float(np.std(mfcc[i]))
        row_dict[f'mfcc_delta_{i+1}'] = float(np.mean(mfcc_delta[i]))
    for i in range(contrast.shape[0]):
        row_dict[f'contrast_mean_{i+1}'] = float(np.mean(contrast[i]))
    return row_dict

def analyze_audio(y, sr, model, feature_cols):
    """Menganalisis audio secara keseluruhan dan per windowing segmen 2 detik."""
    win_samples = int(2.0 * sr)
    hop_samples = int(1.0 * sr)

    if len(y) < win_samples:
        y_padded = np.pad(y, (0, win_samples - len(y)), mode='constant')
    else:
        y_padded = y

    segment_probs = []
    segment_times = []

    for start in range(0, len(y_padded) - win_samples + 1, hop_samples):
        seg = y_padded[start : start + win_samples]
        row_dict = extract_segment_features(seg, sr=sr)
        feats = [row_dict.get(col, 0.0) for col in feature_cols]
        X_df = pd.DataFrame([feats], columns=feature_cols)
        prob = model.predict_proba(X_df.values)[0]
        segment_probs.append(prob)
        segment_times.append(start / sr)

    if not segment_probs:
        row_dict = extract_segment_features(y_padded[:win_samples], sr=sr)
        feats = [row_dict.get(col, 0.0) for col in feature_cols]
        X_df = pd.DataFrame([feats], columns=feature_cols)
        prob = model.predict_proba(X_df.values)[0]
        segment_probs.append(prob)
        segment_times.append(0.0)

    avg_prob = np.mean(segment_probs, axis=0)
    final_class_idx = int(np.argmax(avg_prob))

    # Fitur akustik representatif audio
    rms_overall = float(np.mean(librosa.feature.rms(y=y)))
    cent_overall = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
    zcr_overall = float(np.mean(librosa.feature.zero_crossing_rate(y=y)))

    return {
        "final_class_idx": final_class_idx,
        "avg_prob": avg_prob,
        "segment_probs": np.array(segment_probs),
        "segment_times": segment_times,
        "metrics": {
            "duration": len(y) / sr,
            "rms": rms_overall,
            "centroid": cent_overall,
            "zcr": zcr_overall
        }
    }

# ==========================================
# INISIALISASI
# ==========================================
model, feature_cols, class_names = load_ml_model()

# ==========================================
# SIDEBAR
# ==========================================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3105/3105807.png", width=64)
    st.title("Proyek Sains Data")
    st.markdown("**Konservasi Air Wudhu Masjid**")
    st.caption("Deteksi Aliran Air Berbasis Machine Learning")
    
    st.divider()
    
    st.subheader("📌 Status Sistem")
    if model is not None:
        st.success(f"Model: Ensemble Siap ({len(feature_cols)} Fitur)")
        st.metric(label="Akurasi Model (Test)", value="97.31%")
        st.metric(label="5-Fold CV Score", value="97.54%")
    else:
        st.error("Model .pkl belum ditemukan di folder output/.")

    st.divider()

    st.subheader("🏷️ Kategori Kelas")
    st.markdown("""
    - ⚪ **Bukan Air**: Kebisingan lingkungan, kipas, adzan, obrolan.
    - 🟢 **Kran Pelan (Hemat)**: Aliran wajar, hemat & ramah lingkungan.
    - 🔴 **Kran Keras (Boros)**: Aliran deras/bertekanan tinggi (boros air).
    """)

    st.divider()
    st.caption("Deploy: Docker Container @ Proxmox Lab Server | Port 8501")

# ==========================================
# HEADER
# ==========================================
st.markdown("""
<div class="main-header">
    <div class="badge-psd">TUGAS BESAR PROYEK SAINS DATA (PSD)</div>
    <h1>🚰 Sistem Cerdas Klasifikasi Aliran Air Tempat Wudhu Masjid</h1>
    <p>Monitoring Akustik & Deteksi Keborosan Air Wudhu Secara Real-Time Berbasis Machine Learning</p>
</div>
""", unsafe_allow_html=True)

# Cek model tersedia
if model is None:
    st.error("File model `output/model_aliran_air.pkl` tidak ditemukan. Pastikan file model sudah ada di folder `output/`.")
    st.stop()

# ==========================================
# TABS UTAMA
# ==========================================
tab_detect, tab_visualize, tab_model_info = st.tabs([
    "🎙️ Deteksi & Klasifikasi Suara",
    "📊 Visualisasi Akustik & Spektrogram",
    "ℹ️ Info Model & Arsitektur"
])

# State penampung audio yang dipilih
if "audio_source" not in st.session_state:
    st.session_state.audio_source = None
if "audio_bytes" not in st.session_state:
    st.session_state.audio_bytes = None
if "audio_name" not in st.session_state:
    st.session_state.audio_name = ""

# ==========================================
# TAB 1: DETEKSI & KLASIFIKASI
# ==========================================
with tab_detect:
    st.subheader("1. Pilih atau Rekam Suara Aliran Air")
    
    input_method = st.radio(
        "Pilih Sumber Audio:",
        ["📁 Upload File Audio", "🎙️ Rekam via Mikrofon Langsung", "⚡ Gunakan Sampel Cepat (Demo)"],
        horizontal=True
    )

    audio_bytes_to_play = None
    audio_path_to_load = None
    y_audio = None
    sr_audio = 22050

    if input_method == "📁 Upload File Audio":
        uploaded_file = st.file_uploader(
            "Unggah rekaman suara kran (Format: WAV, MP3, M4A, OGG, FLAC):",
            type=["wav", "mp3", "m4a", "ogg", "flac"]
        )
        if uploaded_file is not None:
            audio_bytes_to_play = uploaded_file.read()
            st.session_state.audio_bytes = audio_bytes_to_play
            st.session_state.audio_name = uploaded_file.name

    elif input_method == "🎙️ Rekam via Mikrofon Langsung":
        st.write("Klik tombol rekam di bawah ini untuk mengambil suara langsung dari mikrofon perangkat Anda:")
        # Streamlit native audio input widget (Streamlit 1.37+)
        if hasattr(st, "audio_input"):
            recorded_audio = st.audio_input("Rekam Suara Mikrofon")
            if recorded_audio is not None:
                audio_bytes_to_play = recorded_audio.read()
                st.session_state.audio_bytes = audio_bytes_to_play
                st.session_state.audio_name = "Rekaman_Mikrofon_Langsung.wav"
        else:
            st.warning("Versi Streamlit ini belum mendukung widget audio_input native. Gunakan opsi Upload File Audio.")

    elif input_method == "⚡ Gunakan Sampel Cepat (Demo)":
        st.write("Uji coba instan menggunakan sampel suara kran wudhu yang telah disiapkan:")
        c1, c2, c3 = st.columns(3)
        sample_choice = None
        with c1:
            if st.button("🔴 Uji Sampel Kran Keras (Boros)", use_container_width=True):
                sample_choice = "sample_kran_keras.mp3"
        with c2:
            if st.button("🟢 Uji Sampel Kran Pelan (Hemat)", use_container_width=True):
                sample_choice = "sample_kran_pelan.mp3"
        with c3:
            if st.button("⚪ Uji Sampel Bukan Air (Noise)", use_container_width=True):
                sample_choice = "sample_bukan_air.wav"

        if sample_choice:
            sample_full_path = os.path.join(SAMPLES_DIR, sample_choice)
            if os.path.exists(sample_full_path):
                with open(sample_full_path, "rb") as f:
                    audio_bytes_to_play = f.read()
                st.session_state.audio_bytes = audio_bytes_to_play
                st.session_state.audio_name = sample_choice
            else:
                st.error(f"File sampel {sample_choice} tidak ditemukan di folder samples/.")

    # Menampilkan Player Audio jika ada data audio aktif
    if st.session_state.audio_bytes is not None:
        st.divider()
        st.write(f"🎧 **Audio Terpilih:** `{st.session_state.audio_name}`")
        st.audio(st.session_state.audio_bytes)

        btn_analyze = st.button("🚀 Jalankan Analisis & Klasifikasi Model", type="primary", use_container_width=True)

        if btn_analyze or "last_result" in st.session_state:
            # Muat audio menggunakan Librosa
            with st.spinner("Memproses sinyal audio dan mengekstrak 81 fitur akustik..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                    tmp.write(st.session_state.audio_bytes)
                    tmp_path = tmp.name

                try:
                    y_audio, sr_audio = librosa.load(tmp_path, sr=22050)
                    result = analyze_audio(y_audio, sr_audio, model, feature_cols)
                    st.session_state.last_result = result
                    st.session_state.y_audio = y_audio
                    st.session_state.sr_audio = sr_audio
                except Exception as e:
                    st.error(f"Gagal memproses audio: {e}")
                finally:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)

            if "last_result" in st.session_state:
                res = st.session_state.last_result
                pred_idx = res["final_class_idx"]
                pred_name = class_names[pred_idx]
                prob_pct = res["avg_prob"] * 100.0

                st.divider()
                st.subheader("2. Hasil Klasifikasi & Rekomendasi")

                # Banner Hasil
                if pred_idx == 2:  # Kran Keras (Boros)
                    st.markdown("""
                    <div class="alert-card alert-keras">
                        <h3 style="margin-top:0; color:#991b1b;">⚠️ PERINGATAN: ALIRAN AIR TERLALU DERAS (BOROS)!</h3>
                        <p style="font-size:1.1rem; margin-bottom:8px;">Terdeteksi aliran air bertekanan tinggi dengan cipratan deras. Berpotensi membuang puluhan liter air wudhu sia-sia.</p>
                        <b>Tindakan Sistem:</b> Segera putar kran ke debit lebih kecil untuk konservasi air masjid!
                    </div>
                    """, unsafe_allow_html=True)
                elif pred_idx == 1:  # Kran Pelan (Hemat)
                    st.markdown("""
                    <div class="alert-card alert-pelan">
                        <h3 style="margin-top:0; color:#166534;">✅ ALIRAN AIR KRAN PELAN (HEMAT & SESUAI SUNNAH)</h3>
                        <p style="font-size:1.1rem; margin-bottom:8px;">Terdeteksi aliran air laminar wajar dan ramah lingkungan. Debit air ideal untuk berwudhu secara hemat.</p>
                        <b>Tindakan Sistem:</b> Pertahankan debit air ini. Alarm hanya aktif jika kran mengalir > 20 detik tanpa henti (indikasi lupa mematikan kran).
                    </div>
                    """, unsafe_allow_html=True)
                else:  # Bukan Air
                    st.markdown("""
                    <div class="alert-card alert-bukan">
                        <h3 style="margin-top:0; color:#334155;">ℹ️ SUARA BUKAN ALIRAN AIR KRAN</h3>
                        <p style="font-size:1.1rem; margin-bottom:8px;">Terdeteksi suara suasana lingkungan masjid (kipas angin / adzan / percakapan / kebisingan). Kran air sedang tertutup.</p>
                        <b>Tindakan Sistem:</b> Sistem standby. Tidak ada air yang mengalir.
                    </div>
                    """, unsafe_allow_html=True)

                # Ringkasan Metrik Cepat
                m1, m2, m3, m4 = st.columns(4)
                with m1:
                    st.metric("Hasil Prediksi", pred_name.split()[0] + " " + pred_name.split()[1])
                with m2:
                    st.metric("Tingkat Keyakinan", f"{prob_pct[pred_idx]:.2f}%")
                with m3:
                    st.metric("Durasi Audio", f"{res['metrics']['duration']:.2f} detik")
                with m4:
                    st.metric("Centroid Spektral", f"{res['metrics']['centroid']:.0f} Hz")

                # Progress Bar Probabilitas
                st.write("#### 📊 Distribusi Probabilitas Softmax Model:")
                col_p1, col_p2, col_p3 = st.columns(3)
                with col_p1:
                    st.markdown(f"**⚪ Bukan Air / Noise:** `{prob_pct[0]:.2f}%`")
                    st.progress(float(res['avg_prob'][0]))
                with col_p2:
                    st.markdown(f"**🟢 Kran Pelan (Hemat):** `{prob_pct[1]:.2f}%`")
                    st.progress(float(res['avg_prob'][1]))
                with col_p3:
                    st.markdown(f"**🔴 Kran Keras (Boros):** `{prob_pct[2]:.2f}%`")
                    st.progress(float(res['avg_prob'][2]))

                # Prediksi Per Detik (Timeline jika durasi audio > 2 detik)
                if len(res["segment_times"]) > 1:
                    st.write("#### ⏱️ Prediksi Real-Time Sepanjang Waktu (Per Window 2 Detik):")
                    df_timeline = pd.DataFrame(
                        res["segment_probs"] * 100.0,
                        columns=["Bukan Air (%)", "Kran Pelan (%)", "Kran Keras (%)"],
                        index=[f"Detik {t:.1f}" for t in res["segment_times"]]
                    )
                    st.line_chart(df_timeline)

# ==========================================
# TAB 2: VISUALISASI AKUSTIK & SPEKTROGRAM
# ==========================================
with tab_visualize:
    st.subheader("Visualisasi Sinyal Audio & Spektrogram Frekuensi")
    
    if "y_audio" in st.session_state and st.session_state.y_audio is not None:
        y_sig = st.session_state.y_audio
        sr_sig = st.session_state.sr_audio

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
        
        # 1. Waveform
        librosa.display.waveshow(y_sig, sr=sr_sig, ax=ax1, color="#0284c7")
        ax1.set_title("Waveform (Amplitudo vs Waktu)", fontsize=12, fontweight='bold')
        ax1.set_xlabel("Waktu (detik)")
        ax1.set_ylabel("Amplitudo")
        ax1.grid(True, linestyle="--", alpha=0.5)

        # 2. Mel-Spectrogram
        S_mel = librosa.feature.melspectrogram(y=y_sig, sr=sr_sig, n_mels=128, fmax=8000)
        S_dB = librosa.power_to_db(S_mel, ref=np.max)
        img = librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=sr_sig, fmax=8000, ax=ax2, cmap='magma')
        ax2.set_title("Mel-Spectrogram (Pola Energi Frekuensi Suara)", fontsize=12, fontweight='bold')
        ax2.set_xlabel("Waktu (detik)")
        ax2.set_ylabel("Frekuensi (Hz)")
        fig.colorbar(img, ax=ax2, format='%+2.0f dB')

        plt.tight_layout()
        st.pyplot(fig)

        # Info Karakteristik Sinyal
        st.markdown("### Karakteristik Akustik Terukur:")
        met = st.session_state.last_result["metrics"]
        c_a, c_b, c_c = st.columns(3)
        with c_a:
            st.info(f"**RMS Energy:** `{met['rms']:.5f}` (Tingkat kekuatan suara)")
        with c_b:
            st.info(f"**Spectral Centroid:** `{met['centroid']:.1f} Hz` (Kecerahan spektrum)")
        with c_c:
            st.info(f"**Zero Crossing Rate:** `{met['zcr']:.5f}` (Kerapatan perubahan tanda)")
    else:
        st.info("Silakan jalankan deteksi audio terlebih dahulu pada tab pertama untuk memunculkan visualisasi gelombang dan spektrogram.")

# ==========================================
# TAB 3: INFO MODEL & ARSITEKTUR
# ==========================================
with tab_model_info:
    st.subheader("Arsitektur & Metodologi Machine Learning")
    
    st.markdown("""
    Sistem ini dibangun untuk mendeteksi suara kran air wudhu menggunakan pendekatan **Acoustic Pattern Recognition**:
    
    #### 1. Ekstraksi Fitur Akustik (81 Fitur)
    - **Temporal & Energi**: RMS Energy, Zero Crossing Rate (Mean & Std), Spectral Crest, Spectral Flux.
    - **Spektral**: Spectral Centroid, Bandwidth, Spectral Rolloff (85% & 25%), Spectral Flatness.
    - **Harmonis vs Perkusi**: HPSS (Harmonic-Percussive Source Separation) Harmonic Ratio.
    - **Rasio Sub-band Energi**: Frekuensi Rendah (<250 Hz), Sedang (250–3000 Hz), dan Tinggi (>3000 Hz).
    - **Representasi Timbre**: 20 Koefisien MFCC (Mean & Std), 20 Delta MFCC, dan Spectral Contrast.

    #### 2. Model Klasifikasi (Soft-Voting Ensemble)
    Model menggunakan penggabungan dua algoritma pohon keputusan terbaik:
    - **ExtraTrees Classifier**: Mengurangi variansi dan sangat tahan terhadap noise lingkungan tempat wudhu.
    - **Tuned Random Forest**: Mengoptimalkan keputusan batas (decision boundary) antara aliran air pelan dan deras.

    #### 3. Evaluasi Performa
    """)

    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.success("""
        **Data Uji Independen (Test Accuracy): 97.31%**
        - Presisi Kran Hemat: > 96%
        - Presisi Kran Boros: > 98%
        - Presisi Bukan Air: > 97%
        """)
    with col_m2:
        st.info("""
        **5-Fold Stratified Cross-Validation: 97.54% (±0.24%)**
        - Performa stabil di berbagai variasi data suara.
        - Tahan terhadap suara adzan, kipas masjid, dan kebisingan sekitar.
        """)
