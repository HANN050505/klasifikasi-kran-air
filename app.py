"""
app.py
======
Aplikasi Antarmuka Interaktif Streamlit:
Sistem Cerdas Klasifikasi & Deteksi Aliran Air Tempat Wudhu Masjid
Berbasis Machine Learning (Soft-Voting Ensemble ExtraTrees + Tuned Random Forest).

Dapat dijalankan langsung dengan:
    streamlit run app.py
atau:
    python app.py
atau melalui Docker:
    docker compose up --build
"""

import os
import io
import sys
import tempfile
import warnings
import joblib
import numpy as np
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt
import streamlit as st
from streamlit import runtime

warnings.filterwarnings('ignore')

# Jika script dieksekusi langsung dengan 'python app.py', otomatis jalankan Streamlit runner
if __name__ == '__main__' and not runtime.exists():
    import subprocess
    print("[*] Terdeteksi dijalankan via 'python app.py'.")
    print("[*] Membuka aplikasi menggunakan Streamlit runner di http://localhost:8501 ...")
    cmd = [sys.executable, "-m", "streamlit", "run", os.path.abspath(__file__)] + sys.argv[1:]
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        pass
    sys.exit(0)

# ==========================================
# KONFIGURASI HALAMAN STREAMLIT
# ==========================================
st.set_page_config(
    page_title="Sistem Deteksi Aliran Air Tempat Wudhu Masjid",
    page_icon="🚰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (CSS) Modern & Responsif
st.markdown("""
<style>
    /* Styling Header Utama */
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
        color: #ffffff !important;
    }
    .main-header p {
        font-size: 0.98rem;
        margin-bottom: 0px;
        color: #e0f2fe !important;
    }
    .badge-psd {
        background-color: #38bdf8;
        color: #0c4a6e;
        padding: 4px 12px;
        border-radius: 6px;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.5px;
        display: inline-block;
        margin-bottom: 8px;
    }

    /* Kartu Peringatan Hasil Deteksi */
    .alert-card {
        padding: 20px 24px;
        border-radius: 12px;
        margin-top: 15px;
        margin-bottom: 20px;
        font-weight: 500;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
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

    /* Kartu Informasi Ilmiah */
    .method-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 16px 20px;
        margin-bottom: 14px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.02);
    }
    .stat-pill {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.82rem;
        font-weight: 600;
        margin-right: 6px;
    }
    .pill-green { background: #dcfce7; color: #15803d; }
    .pill-blue { background: #e0f2fe; color: #0369a1; }
    .pill-red { background: #fee2e2; color: #b91c1c; }
    .pill-slate { background: #f1f5f9; color: #475569; }

    /* Custom Metric Styling */
    div[data-testid="stMetricValue"] {
        font-size: 1.6rem !important;
        font-weight: 700 !important;
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
    """Ekstraksi 81 fitur akustik multi-domain per segmen jendela 2 detik."""
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

    # Dekomposisi HPSS (Harmonic-Percussive Source Separation)
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

# Inisialisasi model
model, feature_cols, class_names = load_ml_model()

# ==========================================
# SIDEBAR
# ==========================================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3105/3105807.png", width=64)
    st.title("Proyek Sains Data")
    st.markdown("**Konservasi Air Wudhu Masjid**")
    st.caption("Monitoring Akustik & Deteksi Keborosan Aliran Air")
    
    st.divider()
    
    st.subheader("📌 Status Validasi Sistem")
    st.markdown("""
    <div style="background:#f0fdf4; border:1px solid #bbf7d0; border-radius:8px; padding:12px; margin-bottom:12px;">
        <div style="font-size:0.78rem; color:#166534; font-weight:700;">ARSITEKTUR ENSEMBLE:</div>
        <div style="font-size:1.02rem; font-weight:700; color:#15803d;">Soft-Voting (ET + RF)</div>
        <div style="font-size:0.74rem; color:#166534; margin-top:4px;">
            Tervalidasi melalui <b>5-Fold Stratified Group K-Fold</b> berbasis rekaman sumber (Zero Data Leakage).
        </div>
    </div>
    """, unsafe_allow_html=True)

    c_sb1, c_sb2 = st.columns(2)
    with c_sb1:
        st.metric("Test Accuracy", "97.31%")
    with c_sb2:
        st.metric("5-Fold CV", "97.56%")

    st.divider()

    st.subheader("🏷️ Kategori Kelas")
    st.markdown("""
    - <span class="stat-pill pill-slate">⚪ 0: Bukan Air</span><br><small>Kebisingan masjid, adzan, kipas angin, obrolan.</small>
    - <span class="stat-pill pill-green">🟢 1: Kran Pelan</span><br><small>Aliran laminar, hemat air & sesuai sunnah wudhu.</small>
    - <span class="stat-pill pill-red">🔴 2: Kran Keras</span><br><small>Aliran deras turbulen bertekanan tinggi (boros air).</small>
    """, unsafe_allow_html=True)

    st.divider()

    st.subheader("👥 Tim Pengembang (PSD)")
    st.markdown("""
    1. **Raihan Aryanova Narendra** (240411100094)
    2. **Muhammad Farhan** (240411100233)
    3. **Joni Kumar Meghwar** (240411100234)
    """)

    st.divider()
    st.caption("Deploy: Docker Container @ Proxmox Lab Server | Port 8501")

# ==========================================
# HEADER UTAMA
# ==========================================
st.markdown("""
<div class="main-header">
    <div class="badge-psd">TUGAS BESAR PROYEK SAINS DATA (PSD)</div>
    <h1>🚰 Sistem Cerdas Klasifikasi Aliran Air Tempat Wudhu Masjid</h1>
    <p>Monitoring Akustik Real-Time & Rekomendasi Konservasi Air Masjid Berbasis Ensemble Machine Learning</p>
</div>
""", unsafe_allow_html=True)

if model is None:
    st.error("File model `output/model_aliran_air.pkl` tidak ditemukan. Pastikan file model sudah ada di folder `output/`.")
    st.stop()

# ==========================================
# TABS UTAMA APLIKASI
# ==========================================
tab_detect, tab_visualize, tab_model_info = st.tabs([
    "🎙️ Deteksi & Klasifikasi Suara",
    "📊 Visualisasi Akustik & Spektrogram",
    "ℹ️ Metodologi, Arsitektur Model & Validasi Ilmiah"
])

# State penampung audio
if "audio_bytes" not in st.session_state:
    st.session_state.audio_bytes = None
if "audio_name" not in st.session_state:
    st.session_state.audio_name = ""

# ==========================================
# TAB 1: DETEKSI & KLASIFIKASI
# ==========================================
with tab_detect:
    st.markdown("### 1. Masukkan atau Rekam Suara Aliran Air")
    
    input_method = st.radio(
        "Pilih Sumber Audio:",
        ["📁 Upload File Audio", "🎙️ Rekam via Mikrofon Langsung", "⚡ Gunakan Sampel Cepat (Demo Preset)"],
        horizontal=True
    )

    audio_bytes_to_play = None

    if input_method == "📁 Upload File Audio":
        uploaded_file = st.file_uploader(
            "Unggah rekaman audio (WAV, MP3, M4A, OGG, FLAC):",
            type=["wav", "mp3", "m4a", "ogg", "flac"]
        )
        if uploaded_file is not None:
            audio_bytes_to_play = uploaded_file.read()
            st.session_state.audio_bytes = audio_bytes_to_play
            st.session_state.audio_name = uploaded_file.name

    elif input_method == "🎙️ Rekam via Mikrofon Langsung":
        st.write("Klik ikon mikrofon di bawah untuk merekam suara secara langsung dari perangkat Anda:")
        if hasattr(st, "audio_input"):
            recorded_audio = st.audio_input("Rekam Suara Mikrofon")
            if recorded_audio is not None:
                audio_bytes_to_play = recorded_audio.read()
                st.session_state.audio_bytes = audio_bytes_to_play
                st.session_state.audio_name = "Rekaman_Mikrofon_Langsung.wav"
        else:
            st.warning("Fitur audio_input belum tersedia pada versi Streamlit ini. Silakan gunakan opsi Upload File Audio.")

    elif input_method == "⚡ Gunakan Sampel Cepat (Demo Preset)":
        st.write("Uji coba instan dengan satu klik menggunakan sampel audio yang telah disediakan:")
        c1, c2, c3 = st.columns(3)
        sample_choice = None
        with c1:
            if st.button("🔴 Uji Sampel: Kran Keras (Boros)", use_container_width=True):
                sample_choice = "sample_kran_keras.mp3"
        with c2:
            if st.button("🟢 Uji Sampel: Kran Pelan (Hemat)", use_container_width=True):
                sample_choice = "sample_kran_pelan.mp3"
        with c3:
            if st.button("⚪ Uji Sampel: Bukan Air (Noise)", use_container_width=True):
                sample_choice = "sample_bukan_air.wav"

        if sample_choice:
            sample_full_path = os.path.join(SAMPLES_DIR, sample_choice)
            if os.path.exists(sample_full_path):
                with open(sample_full_path, "rb") as f:
                    audio_bytes_to_play = f.read()
                st.session_state.audio_bytes = audio_bytes_to_play
                st.session_state.audio_name = sample_choice
            else:
                st.error(f"File sampel `{sample_choice}` tidak ditemukan di folder `samples/`.")

    # Tampilkan Player Audio jika ada data audio aktif
    if st.session_state.audio_bytes is not None:
        st.markdown("---")
        col_ply1, col_ply2 = st.columns([2, 1])
        with col_ply1:
            st.write(f"🎧 **Audio Terpilih:** `{st.session_state.audio_name}`")
            st.audio(st.session_state.audio_bytes)
        with col_ply2:
            st.write("⚡ **Tindakan:**")
            btn_analyze = st.button("🚀 Jalankan Analisis & Klasifikasi", type="primary", use_container_width=True)

        if btn_analyze or "last_result" in st.session_state:
            if btn_analyze:
                with st.spinner("Memproses sinyal audio dan mengekstrak 81 fitur akustik multi-domain..."):
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

                st.markdown("---")
                st.markdown("### 2. Hasil Klasifikasi & Rekomendasi Konservasi Air")

                # Banner Hasil Dinamis
                if pred_idx == 2:  # Kran Keras (Boros)
                    st.markdown("""
                    <div class="alert-card alert-keras">
                        <h3 style="margin-top:0; color:#991b1b;">⚠️ PERINGATAN: ALIRAN AIR TERLALU DERAS (INDIKASI BOROS)!</h3>
                        <p style="font-size:1.05rem; margin-bottom:8px;">
                            Terdeteksi aliran air bertekanan tinggi dengan turbulensi cipratan deras. Berpotensi membuang hingga 15–18 Liter air per menit secara sia-sia.
                        </p>
                        <b>👉 Rekomendasi Sistem:</b> Putar kran ke debit lebih kecil (cukup aliran tenang). Sistem menyalakan alarm peringatan jika aliran deras dibiarkan menyala > 10 detik!
                    </div>
                    """, unsafe_allow_html=True)
                elif pred_idx == 1:  # Kran Pelan (Hemat)
                    st.markdown("""
                    <div class="alert-card alert-pelan">
                        <h3 style="margin-top:0; color:#166534;">✅ ALIRAN AIR KRAN PELAN (HEMAT & RAMAH LINGKUNGAN)</h3>
                        <p style="font-size:1.05rem; margin-bottom:8px;">
                            Terdeteksi aliran air laminar wajar dan stabil (~4–6 Liter/menit). Sesuai dengan anjuran hemat air dan sunnah berwudhu.
                        </p>
                        <b>👉 Rekomendasi Sistem:</b> Pertahankan debit aliran ini. Peringatan hanya diaktifkan jika kran lupa ditutup dan terus mengalir lebih dari 20 detik.
                    </div>
                    """, unsafe_allow_html=True)
                else:  # Bukan Air
                    st.markdown("""
                    <div class="alert-card alert-bukan">
                        <h3 style="margin-top:0; color:#334155;">ℹ️ SUARA LINGKUNGAN (BUKAN ALIRAN AIR KRAN)</h3>
                        <p style="font-size:1.05rem; margin-bottom:8px;">
                            Terdeteksi suara kebisingan latar masjid (kumandang adzan, kipas angin, langkah kaki, atau percakapan jamaah). Kran air saat ini tertutup.
                        </p>
                        <b>👉 Status Sistem:</b> Standby monitoring. Tidak ada pemborosan air terdeteksi.
                    </div>
                    """, unsafe_allow_html=True)

                # Ringkasan Metrik
                m1, m2, m3, m4, m5 = st.columns(5)
                with m1:
                    st.metric("Kondisi Terdeteksi", pred_name.split()[0] + " " + pred_name.split()[1])
                with m2:
                    st.metric("Keyakinan Model", f"{prob_pct[pred_idx]:.2f}%")
                with m3:
                    st.metric("Durasi Audio", f"{res['metrics']['duration']:.2f} detik")
                with m4:
                    st.metric("Spectral Centroid", f"{res['metrics']['centroid']:.0f} Hz")
                with m5:
                    st.metric("RMS Energy", f"{res['metrics']['rms']:.4f}")

                # Distribusi Probabilitas Tiga Kelas
                st.markdown("#### 📊 Distribusi Probabilitas Softmax Model:")
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

                # Deteksi Timeline Dinamis jika durasi > 2 detik
                if len(res["segment_times"]) > 1:
                    st.markdown("#### ⏱️ Deteksi Dinamis Sepanjang Waktu (Per Interval Jendela 2 Detik):")
                    df_timeline = pd.DataFrame(
                        res["segment_probs"] * 100.0,
                        columns=["Bukan Air (%)", "Kran Pelan (%)", "Kran Keras (%)"],
                        index=[f"Detik {t:.1f}" for t in res["segment_times"]]
                    )
                    st.line_chart(df_timeline)

                # ==========================================
                # SIMULATOR DAMPAK KONSERVASI AIR
                # ==========================================
                st.markdown("---")
                st.markdown("### 💧 Simulator Pemborosan & Potensi Konservasi Air Masjid")
                st.write("Gunakan slider di bawah untuk mensimulasikan dampak jika kondisi kran ini dibiarkan mengalir:")

                dur_slider = st.slider(
                    "Simulasi Durasi Kran Menyala (detik):",
                    min_value=5, max_value=60, value=25, step=5
                )

                if pred_idx == 2:
                    debit_rate = 0.25
                    status_warning = "🚨 MELEBIHI AMBANG KRAN KERAS (10 Detik)! Alarm Pemborosan Aktif!" if dur_slider > 10 else "⚠️ Aliran deras bertekanan tinggi."
                    warn_color = "🔴 Keras (Boros)"
                elif pred_idx == 1:
                    debit_rate = 0.083
                    status_warning = "⚠️ MELEBIHI AMBANG KRAN PELAN (20 Detik)! Terindikasi lupa mematikan kran." if dur_slider > 20 else "✅ Aliran stabil & hemat."
                    warn_color = "🟢 Pelan (Hemat)" if dur_slider <= 20 else "🟠 Peringatan Lupa"
                else:
                    debit_rate = 0.0
                    status_warning = "⚪ Kran tertutup. Tidak ada air yang mengalir."
                    warn_color = "⚪ Standby"

                volume_terbuang = debit_rate * dur_slider
                sunnah_mud_equivalents = volume_terbuang / 0.75 if volume_terbuang > 0 else 0
                hemat_harian_liter = (0.25 - 0.083) * 20 * 200

                c_sim1, c_sim2, c_sim3, c_sim4 = st.columns(4)
                with c_sim1:
                    st.metric("Air Mengalir", f"{volume_terbuang:.2f} Liter")
                with c_sim2:
                    st.metric("Status Operasional", f"{warn_color}")
                with c_sim3:
                    st.metric("Setara Porsi Wudhu Sunnah", f"~{sunnah_mud_equivalents:.1f}x Wudhu")
                with c_sim4:
                    st.metric("Potensi Hemat (200 Jamaah)", f"{hemat_harian_liter:.0f} Liter/hari")

                st.caption(f"**Evaluasi Ambang Batas Durasi:** {status_warning}")

# ==========================================
# TAB 2: VISUALISASI AKUSTIK & SPEKTROGRAM
# ==========================================
with tab_visualize:
    st.markdown("### Visualisasi Sinyal Audio & Spektrogram Frekuensi")
    
    if "y_audio" in st.session_state and st.session_state.y_audio is not None:
        y_sig = st.session_state.y_audio
        sr_sig = st.session_state.sr_audio

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6.5))
        
        # 1. Waveform Plot
        librosa.display.waveshow(y_sig, sr=sr_sig, ax=ax1, color="#0284c7")
        ax1.set_title("1. Waveform Domain Waktu (Amplitudo vs Waktu)", fontsize=11, fontweight='bold')
        ax1.set_xlabel("Waktu (detik)")
        ax1.set_ylabel("Amplitudo")
        ax1.grid(True, linestyle="--", alpha=0.5)

        # 2. Mel-Spectrogram Plot
        S_mel = librosa.feature.melspectrogram(y=y_sig, sr=sr_sig, n_mels=128, fmax=8000)
        S_dB = librosa.power_to_db(S_mel, ref=np.max)
        img = librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=sr_sig, fmax=8000, ax=ax2, cmap='magma')
        ax2.set_title("2. Mel-Spectrogram (Pola Persebaran Energi Frekuensi 0–8000 Hz)", fontsize=11, fontweight='bold')
        ax2.set_xlabel("Waktu (detik)")
        ax2.set_ylabel("Frekuensi Mel (Hz)")
        fig.colorbar(img, ax=ax2, format='%+2.0f dB')

        plt.tight_layout()
        st.pyplot(fig)

        # Metrik Akustik Terukur
        st.markdown("#### Karakteristik Akustik Terukur:")
        met = st.session_state.last_result["metrics"]
        c_a, c_b, c_c = st.columns(3)
        with c_a:
            st.info(f"**RMS Energy:** `{met['rms']:.5f}` (Tingkat intensitas suara)")
        with c_b:
            st.info(f"**Spectral Centroid:** `{met['centroid']:.1f} Hz` (Frekuensi cipratan air)")
        with c_c:
            st.info(f"**Zero Crossing Rate:** `{met['zcr']:.5f}` (Turbulensi getaran)")
    else:
        st.info("💡 Silakan jalankan analisis audio pada tab pertama terlebih dahulu untuk memunculkan visualisasi gelombang dan spektrogram.")

# ==========================================
# TAB 3: INFO MODEL & METODOLOGI ILMIAH
# ==========================================
with tab_model_info:
    st.markdown("### Metodologi Ilmiah, Rekayasa Fitur Akustik & Validasi Model")

    st.markdown("""
    Sistem deteksi akustik aliran air wudhu ini dibangun dengan metodologi sains data yang ketat untuk memastikan keandalan deteksi di lingkungan nyata tempat wudhu masjid yang dinamis dan berderau.
    """)

    # 4 Pilar Metodologi
    col_p1, col_p2 = st.columns(2)
    with col_p1:
        st.markdown("""
        <div class="method-card">
            <h4 style="color:#0369a1; margin-top:0;">1. Tantangan Fisik Sinyal Akustik Fluida</h4>
            <p style="font-size:0.92rem; color:#334155;">
                Suara air wudhu merupakan derau acak pita lebar (<i>broadband stochastic noise</i>). Aliran laminar (kran pelan) dan aliran turbulen (kran deras) memiliki spektrum dasar yang saling bertumpuk. Perbedaan utamanya terletak pada <b>turbulensi cipratan frekuensi tinggi</b>, pergeseran <i>spectral centroid</i>, serta dinamika fluktuasi energi transien (<i>spectral flux</i>).
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="method-card">
            <h4 style="color:#0369a1; margin-top:0;">2. Kekebalan terhadap Derau Lingkungan Masjid</h4>
            <p style="font-size:0.92rem; color:#334155;">
                Tempat wudhu memiliki gema ubin keramik yang kuat, suara desis bilah kipas angin (<250 Hz), dan kumandang adzan / vokal jamaah. Untuk mencegah <i>false alarm</i>, sistem menerapkan fitur <b>HPSS Harmonic Ratio</b> (memisahkan suara bernada vokal dari air) dan melatih model menggunakan data augmentasi sintesis pada SNR 12–18 dB.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col_p2:
        st.markdown("""
        <div class="method-card">
            <h4 style="color:#0369a1; margin-top:0;">3. Protokol Validasi Bebas Kebocoran Data</h4>
            <p style="font-size:0.92rem; color:#334155;">
                Pemisahan data acak biasa dilarang karena segmen-segmen dari satu rekaman memiliki autokorelasi tinggi (<i>data leakage</i>). Sistem menggunakan <b>5-Fold Stratified Group K-Fold</b> berdasarkan rekaman sumber (<code>file_id</code>). Seluruh segmen dari satu file audio hanya diuji pada model yang belum pernah mendengar suara rekaman tersebut sebelumnya.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="method-card">
            <h4 style="color:#0369a1; margin-top:0;">4. Arsitektur Soft-Voting Ensemble</h4>
            <p style="font-size:0.92rem; color:#334155;">
                Model menggabungkan <b>ExtraTreesClassifier</b> (600 pohon) yang merandomisasi batas pemisahan untuk meminimalkan variansi terhadap derau lingkungan, dipadukan dengan <b>Tuned RandomForestClassifier</b> (400 pohon) untuk mengoptimalkan margin pemisahan kelas melalui rata-rata probabilitas halus (<i>soft voting</i>).
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("#### 📊 Spesifikasi Teknis & Hasil Pengujian Komprehensif:")
    df_spec = pd.DataFrame({
        "Komponen Metodologi": [
            "Jumlah Rekaman Sumber (Ground Truth)",
            "Jumlah Segmen Audio Jendela 2 Detik",
            "Domain Fitur yang Diekstraksi",
            "Skema Validasi Silang (Cross-Validation)",
            "Akurasi Evaluasi Level Jendela (Window-level)",
            "Akurasi Evaluasi Level Rekaman (Record-level Soft Voting)",
            "Presisi Kelas 'Bukan Air' (Ketahanan False Alarm)",
            "Recall Kelas 'Kran Keras' (Deteksi Aliran Boros)",
            "Macro F1-Score Keseluruhan"
        ],
        "Detail Implementasi & Metrik": [
            "2.092 rekaman sumber unik (529 Bukan Air, 645 Pelan, 918 Keras)",
            "9.660 segmen audio (durasi 2 detik, hop 1 detik)",
            "81 Fitur Multi-Domain (Spektral, Temporal, HPSS, Sub-band, MFCC, Contrast)",
            "5-Fold Stratified Group K-Fold (Strict Grouping by file_id)",
            "95.23% – 95.55%",
            "97.42% – 97.56% (2.038 dari 2.092 rekaman benar)",
            "99.80% (Sistem hampir tidak pernah memicu alarm palsu pada adzan/kipas)",
            "98.80% (Aliran deras yang boros hampir selalu berhasil terdeteksi)",
            "97.50%"
        ]
    })
    st.dataframe(df_spec, use_container_width=True, hide_index=True)

    st.divider()

    # Tampilkan Grafik Evaluasi
    plot_eval_path = os.path.join(BASE_DIR, "output", "evaluasi_model.png")
    if os.path.exists(plot_eval_path):
        st.markdown("#### 🖼️ Visualisasi Evaluasi Model (Confusion Matrix, Top Fitur & Distribusi Data):")
        st.image(plot_eval_path, use_container_width=True, caption="Evaluasi Confusion Matrix, Feature Importance, dan Distribusi Segmen Kelas")
    
    st.divider()

    st.markdown("""
    #### 🔬 Rincian 81 Fitur Akustik Multi-Domain:
    1. **Karakteristik Spektral Dasar (6 Fitur):**
       - `rms`: Root Mean Square energy (amplitudo kekuatan energi akustik).
       - `spectral_centroid`: Titik pusat massa frekuensi (tinggi pada cipratan deras).
       - `spectral_bandwidth`: Lebar rentang frekuensi aliran fluida.
       - `spectral_rolloff` (85% & 25%): Frekuensi cutoff akumulasi energi sinyal.
       - `spectral_flatness`: Tingkat kemiripan sinyal dengan white noise acak.
    2. **Dinamika Temporal & Transien (4 Fitur):**
       - `zcr`: Zero Crossing Rate rata-rata (fluktuasi perlintasan nol).
       - `zcr_std`: Standar deviasi ZCR (menangkap ketidakteraturan tetesan air).
       - `crest`: Crest factor (rasio puncak terhadap rata-rata energi).
       - `flux`: Spectral flux onset strength (laju perubahan spektrum antar-frame).
    3. **Dekomposisi Sumber Harmonik (1 Fitur):**
       - `harm_ratio`: Rasio energi harmonik via HPSS (krusial menyaring suara vokal kumandang adzan dari gemercik air).
    4. **Proporsi Energi Sub-band (3 Fitur):**
       - `low_ratio`: Frekuensi rendah (<250 Hz, mencirikan derau putaran mesin/kipas angin).
       - `mid_ratio`: Frekuensi menengah (250–3000 Hz, pita frekuensi utama aliran air).
       - `high_ratio`: Frekuensi tinggi (>3000 Hz, mencirikan cipratan air bertekanan tinggi).
    5. **Tekstur Timbre MFCC (60 Fitur):**
       - 20 Koefisien MFCC Mean (representasi bentuk formant spektral).
       - 20 Koefisien MFCC Std (variabilitas temporal timbre).
       - 20 Koefisien MFCC Delta Mean (laju perubahan tekstur suara).
    6. **Spectral Contrast (7 Fitur):**
       - 7 Sub-band Contrast Mean (kedalaman puncak vs lembah pada 7 pita spektrum).
    """)
