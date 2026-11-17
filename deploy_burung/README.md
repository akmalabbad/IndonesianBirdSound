# Rimba — dashboard pengenalan suara burung

Folder ini sudah mandiri: kode, model, statistik lengkap, dan contoh audio tersedia.
Tidak perlu membawa notebook, dataset lengkap, atau proyek Hijaiyah.

## Upload ke GitHub

1. Buat repository baru di GitHub.
2. Upload **isi folder `deploy_burung`** ke bagian utama repository, bukan folder
   induk proyek penelitian. Sertakan `.streamlit` dan `.gitignore` yang tersembunyi.
3. Pastikan `requirements.txt`, `dashboard`, `src`, `models`, `data`, `outputs`,
   dan `sample_audio` berada pada tingkat yang sama.

```text
requirements.txt
README.md
.gitignore
.streamlit/config.toml
dashboard/app.py
dashboard/data_access.py
dashboard/__init__.py
dashboard/assets/forest.svg
src/audio_utils.py
models/best_model.joblib
data/processed/valid_audio.csv
outputs/metrics/...
sample_audio/manifest.csv
sample_audio/ATTRIBUTION.csv
sample_audio/*.mp3
```

## Deploy ke Streamlit Community Cloud

1. Masuk ke https://share.streamlit.io/ dengan akun GitHub.
2. Pilih **Create app**, lalu repository dan branch yang berisi file di atas.
3. Isi **Main file path** dengan `dashboard/app.py`.
4. Pada **Advanced settings**, pilih **Python 3.11**, lalu deploy.

Dependensi sudah dicantumkan pada `requirements.txt` di root. Versi scikit-learn
dibuat sama dengan versi saat model disimpan. Aplikasi membaca lokasi file dari
lokasi kodenya dan mendukung pemisah path Windows maupun Linux.

Panduan resmi:
- https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy
- https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies

## Mencoba secara lokal

Buka terminal di folder ini, lalu:

```powershell
python -m pip install -r requirements.txt
python -m streamlit run dashboard/app.py
```

## Data dan contoh rekaman

Statistik dan skor evaluasi tetap berasal dari seluruh eksperimen: 1.919 rekaman,
26 spesies, dan 421 rekaman test. Paket hanya menyertakan satu contoh test per
spesies. Contoh dipilih berdasarkan ukuran file kecil, bukan ketepatan prediksi.
Hasil contoh ini tidak menggantikan skor test penuh.

Rekaman disalin utuh tanpa pemotongan atau pengubahan encoding. Atribusi nama
perekam, tautan sumber, dan lisensi masing-masing rekaman tersedia dalam
`sample_audio/ATTRIBUTION.csv` dan pada pemutar audio. Hak rekaman tetap pada
pemiliknya; sebagian contoh memakai lisensi nonkomersial (NC) dan/atau melarang
turunan (ND). Ikuti lisensi per rekaman ketika menggunakan ulang paket.

Prediksi memakai maksimal 30 detik pertama, sama dengan pelatihan. Pengguna juga
bisa mengunggah audio sendiri. Unggahan diproses sementara pada server dan tidak
ditambahkan ke repository. Model hanya mengenali 26 spesies dan skor belum
dikalibrasi; Macro F1 test sekitar 50,3%.

Paket ini sudah diuji lokal tanpa dataset lengkap. Deploy ke akun GitHub dan
Streamlit dilakukan oleh pemilik akun; pengujian lokal belum membuktikan deploy
di server berhasil.
