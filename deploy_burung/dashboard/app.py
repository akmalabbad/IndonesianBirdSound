"""Dashboard suara burung. Jalankan dari root: python -m streamlit run dashboard/app.py."""
from pathlib import Path
import base64
import html
import json
import sys
import tempfile

import joblib
import librosa
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.audio_utils import extract_features, load_audio
from dashboard.data_access import resolve_audio, available_paths

st.set_page_config(page_title="Rimba · Suara Burung Indonesia", page_icon="🌿", layout="wide")
GREEN = "#35765B"


@st.cache_data
def read_data():
    audio = pd.read_csv(ROOT / "data/processed/valid_audio.csv")
    experiment = json.loads((ROOT / "outputs/metrics/experiment.json").read_text())
    return audio, experiment


@st.cache_resource
def read_model():
    return joblib.load(ROOT / "models/best_model.joblib")


@st.cache_data
def playable_paths(paths):
    return available_paths(ROOT, paths)


def metric_file(name, **kwargs):
    return pd.read_csv(ROOT / "outputs/metrics" / name, **kwargs)


def chart(fig, height=350):
    fig.update_layout(height=height, margin=dict(l=10, r=15, t=30, b=10),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(color="#254b3b"), colorway=[GREEN, "#c6a15b", "#8caa91"])
    st.plotly_chart(fig, width="stretch")


def audio_path(relative):
    path = resolve_audio(ROOT, relative)
    if path is None:
        raise ValueError("Audio ini tidak disertakan dalam paket dashboard.")
    return path


def audio_plots(path):
    config = read_model()["config"]
    y, sr = load_audio(path, config)
    wave, mel = st.tabs(["Gelombang suara", "Mel spectrogram"])
    with wave:
        step = max(1, len(y) // 10000)
        fig = go.Figure(go.Scatter(x=np.arange(0, len(y), step) / sr,
                                  y=y[::step], line=dict(color=GREEN, width=1)))
        fig.update_layout(xaxis_title="Waktu (detik)", yaxis_title="Amplitudo")
        chart(fig, 250)
    with mel:
        spectrum = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64)
        db = librosa.power_to_db(spectrum, ref=np.max)
        fig = go.Figure(go.Heatmap(z=db, x=librosa.frames_to_time(np.arange(db.shape[1]), sr=sr),
                                  y=np.arange(1, 65), colorscale="Viridis", colorbar_title="dB"))
        fig.update_layout(xaxis_title="Waktu (detik)", yaxis_title="Pita mel")
        chart(fig, 270)
    st.caption(f"Visualisasi memakai maksimal {config['max_duration']} detik pertama, mono {sr:,} Hz.")


forest = base64.b64encode((Path(__file__).parent / "assets/forest.svg").read_bytes()).decode()
st.markdown("""<style>
.block-container{padding-top:2rem;max-width:1400px;padding-bottom:3rem}
[data-testid="stSidebar"]{border-right:1px solid #dce2d6}
h1,h2,h3{letter-spacing:-.035em}h1{font-size:3rem!important}
[data-testid="stMetric"]{background:#fffef8;padding:20px;border:1px solid #e1e5d9;border-radius:16px}
[data-testid="stMetricLabel"]{color:#65816c}[data-testid="stMetricValue"]{font-size:2rem}
.eyebrow{font-size:.72rem;letter-spacing:.18em;font-weight:700;color:#a9c9ae;margin-bottom:18px}
.hero{border-radius:24px;padding:45px 42px;color:#f8f6e9;min-height:310px;background-size:cover;background-position:center;box-shadow:0 12px 30px #153c3012;margin-bottom:25px}
.hero h1{color:#fff9e9;max-width:650px;line-height:1.12;margin:0 0 16px}.hero p{max-width:490px;color:#d4e0d1;line-height:1.7;font-size:1rem}.tag{display:inline-block;padding:6px 12px;border:1px solid #b0c6a855;border-radius:30px;font-size:.75rem;color:#dce8d5}
.note{padding:20px 24px;background:#e9eee1;border-left:3px solid #7a9a65;border-radius:0 14px 14px 0;line-height:1.7;margin:15px 0}
.prediction{background:#1e4939;border-radius:20px;padding:28px;color:#f9f5df;margin:18px 0}.prediction h2{color:#f9f5df;margin:8px 0}.prediction p{color:#b9d0b8;margin:0}
.footer{border-top:1px solid #dce2d6;margin-top:40px;padding-top:18px;color:#738373;font-size:.78rem}
@media(max-width:700px){.hero{padding:28px 24px}.hero h1{font-size:2.1rem!important}.block-container{padding-top:1rem}}
</style>""", unsafe_allow_html=True)

needed = ["data/processed/valid_audio.csv", "outputs/metrics/experiment.json", "models/best_model.joblib"]
missing = [name for name in needed if not (ROOT / name).exists()]
if missing:
    st.error("Hasil eksperimen belum lengkap. Jalankan notebook Indonesian_Bird_Recognition.ipynb dahulu.")
    st.code("\n".join(missing))
    st.stop()

audio, experiment = read_data()
available = playable_paths(audio.path.tolist())
names = audio.drop_duplicates("species").set_index("species")["english_name"].fillna("").to_dict()
with st.sidebar:
    st.markdown("## 🌿 rimba")
    st.caption("SUARA BURUNG INDONESIA")
    st.divider()
    page = st.radio("Jelajahi", ["Ringkasan", "Jelajah rekaman", "Prediksi audio", "Evaluasi model"])
    st.divider()
    st.markdown("**Dari suara, mengenal alam.**")
    st.caption("Eksperimen pengenalan 26 spesies burung dari rekaman suara.")
    st.caption(f"Model aktif · {experiment['best_model']}")
    st.caption(f"Audio yang bisa diputar: {len(available):,} dari {len(audio):,} rekaman.")

if page == "Ringkasan":
    st.markdown(f'<div class="hero" style="background-image:url(data:image/svg+xml;base64,{forest})">'
                '<div class="eyebrow">CATATAN SUARA DARI ALAM</div><h1>Dengarkan hutan.<br>Kenali penghuninya.</h1>'
                '<p>Menjelajahi keragaman suara burung Indonesia melalui rekaman, ciri audio, dan machine learning.</p>'
                '<span class="tag">26 spesies · rekaman nyata · eksplorasi audio</span></div>', unsafe_allow_html=True)
    cols = st.columns(4)
    for col, label, value in zip(cols, ["Rekaman dalam dataset", "Spesies burung", "Total durasi dataset", "Macro F1 · test"],
                                  [f"{len(audio):,}", audio.species.nunique(), f"{audio.duration.sum()/3600:.1f} jam", f"{experiment['test_metrics']['macro_f1']:.1%}"]):
        col.metric(label, value)
    if len(available) < len(audio):
        st.caption(f"Statistik memakai seluruh metadata dataset. Versi ini menyediakan {len(available)} contoh audio untuk diputar; unggah audio sendiri tetap tersedia.")
    left, right = st.columns([1.5, 1], gap="large")
    with left:
        st.subheader("Suara yang terkumpul")
        st.caption("Jumlah rekaman untuk setiap spesies dalam dataset lokal.")
        counts = audio.species.value_counts().rename_axis("Spesies").reset_index(name="Rekaman")
        chart(px.bar(counts.sort_values("Rekaman"), x="Rekaman", y="Spesies", orientation="h", color_discrete_sequence=[GREEN]), 660)
    with right:
        st.subheader("Mulai dari satu rekaman")
        st.write("Pilih **Jelajah rekaman** untuk mendengar contoh suara, atau **Prediksi audio** untuk mencoba rekaman sendiri.")
        st.markdown('<div class="note">Setiap rekaman diringkas menjadi <b>146 fitur audio</b>. Model membaca pola frekuensi, warna suara, dan perubahan bunyi dari maksimal 30 detik pertama.</div>', unsafe_allow_html=True)
        st.subheader("Pembagian data")
        splits = pd.DataFrame(experiment["split_counts"].items(), columns=["Bagian", "Rekaman"])
        chart(px.pie(splits, names="Bagian", values="Rekaman", hole=.68,
                     color_discrete_sequence=[GREEN, "#b7c9a1", "#d6b77b"]), 270)
        st.caption("Rekaman dikelompokkan menurut sesi (perekam, tanggal, lokasi) dan file identik sebelum dibagi.")
    st.caption("Ilustrasi dekoratif hutan dan burung; bukan foto identifikasi spesies.")

elif page == "Jelajah rekaman":
    st.caption("KOLEKSI AUDIO / DATASET LOKAL")
    st.title("Satu spesies, banyak cerita suara.")
    species = st.selectbox("Pilih spesies", sorted(names), format_func=lambda s: f"{names[s]} · {s}")
    subset = audio[audio.species == species]
    a, b, c = st.columns(3)
    a.metric("Rekaman", len(subset)); b.metric("Median durasi", f"{subset.duration.median():.1f} detik")
    c.metric("Negara di metadata", subset.country.nunique())
    playable = subset[subset.path.isin(available)]
    st.caption(f"Metadata: {len(subset)} rekaman · Audio yang disertakan: {len(playable)}.")
    if not playable.empty:
        chosen = st.selectbox("Pilih rekaman", playable.index,
                          format_func=lambda i: f"XC{audio.loc[i, 'id']} · {audio.loc[i, 'location']}")
        row = audio.loc[chosen]
        st.subheader(names[species])
        st.caption(species)
        path = audio_path(row.path)
        st.audio(str(path))
        st.caption(f"Perekam: {row.recordist} · Lokasi: {row.location} · Tanggal: {row.date} · Kualitas: {row.quality}")
        if pd.notna(row.url):
            st.link_button("Lihat rekaman sumber ↗", str(row.url))
        st.caption(f"Lisensi rekaman: {row.license}")
        with st.spinner("Membaca pola suara..."):
            audio_plots(path)
    else:
        st.info("Contoh audio untuk spesies ini belum disertakan. Metadata tetap dapat dilihat di bawah.")
    with st.expander("Daftar rekaman spesies ini"):
        table = subset[["id", "english_name", "location", "duration", "quality", "recordist"]]
        st.dataframe(table, hide_index=True, width="stretch")
        st.download_button("Unduh metadata pilihan", table.to_csv(index=False).encode(), "metadata_spesies.csv", "text/csv")

elif page == "Prediksi audio":
    st.caption("LAB AUDIO / COBA MODEL")
    st.title("Suara burung apa itu?")
    st.write("Unggah rekaman atau gunakan contoh dari data test. Model akan menampilkan tiga kandidat spesies.")
    mode = st.radio("Sumber audio", ["Unggah rekaman", "Contoh data test"], horizontal=True)
    content = None
    suffix = ".wav"
    if mode == "Unggah rekaman":
        uploaded = st.file_uploader("Audio WAV, MP3, FLAC, atau OGG · maksimal 30 MB", type=["wav", "mp3", "flac", "ogg"])
        if uploaded:
            content = uploaded.getvalue()
            suffix = Path(uploaded.name).suffix.lower()
            st.audio(content)
    else:
        examples = metric_file("test_predictions.csv")
        examples = examples[examples.path.isin(available)].reset_index(drop=True)
        if examples.empty:
            st.info("Belum ada contoh audio test yang disertakan. Gunakan pilihan Unggah rekaman.")
        else:
            index = st.selectbox("Pilih contoh", examples.index,
                             format_func=lambda i: f"{examples.loc[i, 'species']} · {Path(examples.loc[i, 'path']).stem}")
            sample_path = audio_path(examples.loc[index, "path"])
            content, suffix = sample_path.read_bytes(), sample_path.suffix
            st.audio(content)
            st.caption(f"Label asli: {examples.loc[index, 'species']}. Ini adalah rekaman test dari eksperimen tersimpan.")
    st.caption("Model selalu memilih salah satu dari 26 kelas. Suara selain burung atau spesies di luar dataset belum dapat ditolak otomatis.")
    if st.button("Kenali suara →", type="primary", disabled=content is None):
        try:
            with tempfile.TemporaryDirectory(prefix="rimba_") as folder:
                path = Path(folder) / f"audio{suffix}"
                path.write_bytes(content)
                with st.spinner("Mengambil ciri suara dan membaca model..."):
                    bundle = read_model()
                    features = extract_features(path, bundle["config"])
                    X = pd.DataFrame([features])[bundle["features"]]
                    model, encoder = bundle["model"], bundle["encoder"]
                    scores = model.predict_proba(X)[0]
                    order = np.argsort(scores)[::-1][:3]
                    labels = encoder.inverse_transform(model.classes_[order].astype(int))
                best = str(labels[0])
                st.markdown(f'<div class="prediction"><p>KANDIDAT UTAMA</p><h2>{html.escape(names.get(best, best))}</h2><p><i>{html.escape(best)}</i></p></div>', unsafe_allow_html=True)
                ranked = pd.DataFrame({"Spesies": labels, "Skor model": scores[order]})
                fig = px.bar(ranked.iloc[::-1], x="Skor model", y="Spesies", orientation="h", text="Skor model", color_discrete_sequence=[GREEN])
                fig.update_traces(texttemplate="%{x:.1%}", textposition="outside")
                fig.update_xaxes(range=[0, 1], tickformat=".0%")
                chart(fig, 260)
                st.info("Skor ini adalah probabilitas model yang belum dikalibrasi, sehingga tidak sama dengan kepastian identifikasi.")
                audio_plots(path)
                st.download_button("Unduh hasil prediksi", ranked.to_csv(index=False).encode(), "prediksi_burung.csv", "text/csv")
        except Exception as error:
            st.error(f"Audio belum bisa dianalisis: {error}")
            st.caption("Coba file audio lain yang berisi suara jelas, tidak kosong, dan tidak rusak.")

else:
    st.caption("HASIL EKSPERIMEN / EVALUASI")
    st.title("Seberapa baik model mengenali suara?")
    st.write(f"Model terpilih: **{experiment['best_model']}**, berdasarkan Macro F1 validation.")
    columns = st.columns(3)
    for col, key, label in zip(columns, ["accuracy", "macro_f1", "balanced_accuracy"], ["Accuracy · test", "Macro F1 · test", "Balanced accuracy · test"]):
        col.metric(label, f"{experiment['test_metrics'][key]:.1%}")
    low, high = experiment["bootstrap_macro_f1_95"]
    st.caption(f"Interval bootstrap kelompok 95% untuk Macro F1 test: {low:.1%}–{high:.1%}. Jumlah rekaman test: {experiment['split_counts']['test']}.")
    comparison, classes, confusion, clusters = st.tabs(["Perbandingan model", "Per spesies", "Confusion matrix", "Clustering"])
    with comparison:
        scores = metric_file("model_comparison.csv")
        st.caption("Semua skor pada grafik ini berasal dari validation. Data test hanya digunakan untuk evaluasi akhir model terpilih.")
        fig = px.bar(scores.sort_values("macro_f1"), x="macro_f1", y="model", orientation="h", color_discrete_sequence=[GREEN])
        fig.update_xaxes(range=[0, 1], tickformat=".0%", title="Macro F1 · validation")
        chart(fig, 390)
        st.dataframe(scores, hide_index=True, width="stretch")
    with classes:
        report = metric_file("test_classification_report.csv", index_col=0).loc[sorted(names)]
        fig = px.bar(report.sort_values("f1-score").reset_index(), x="f1-score", y=report.index.name or "index", orientation="h", color_discrete_sequence=[GREEN])
        fig.update_xaxes(range=[0, 1], tickformat=".0%", title="F1 · test")
        chart(fig, 680)
        st.dataframe(report, width="stretch")
        st.subheader("Pasangan yang sering tertukar")
        st.dataframe(metric_file("confused_species.csv").head(10), hide_index=True, width="stretch")
    with confusion:
        matrix = metric_file("test_confusion_matrix.csv", index_col=0)
        normalize = st.toggle("Tampilkan persentase per label asli", value=True)
        values = matrix.div(matrix.sum(axis=1).replace(0, np.nan), axis=0) if normalize else matrix
        fig = px.imshow(values, color_continuous_scale="Greens", aspect="auto", labels=dict(x="Prediksi", y="Label asli", color="Proporsi" if normalize else "Rekaman"))
        chart(fig, 780)
        st.caption("Baris menunjukkan label asli, kolom menunjukkan prediksi. Prediksi benar berada pada diagonal.")
    with clusters:
        st.write("Clustering mengelompokkan kemiripan fitur tanpa memakai label spesies saat pelatihan.")
        result = metric_file("clustering_comparison.csv")
        st.dataframe(result, hide_index=True, width="stretch")
        st.caption("Silhouette, ARI, NMI, dan Calinski–Harabasz lebih besar lebih baik; Davies–Bouldin lebih kecil lebih baik. Kelompok fitur tidak selalu sama dengan spesies.")
    st.download_button("Unduh ringkasan eksperimen", json.dumps(experiment, indent=2).encode(), "eksperimen_burung.json", "application/json")

st.markdown('<div class="footer">RIMBA / Indonesian Bird Sounds &nbsp; · &nbsp; Eksplorasi audio dan hasil machine learning &nbsp; · &nbsp; Data lokal, hasil eksperimen asli.</div>', unsafe_allow_html=True)
