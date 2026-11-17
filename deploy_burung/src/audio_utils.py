"""Fungsi audio yang dipakai saat latihan dan prediksi."""
from pathlib import Path
import hashlib
import os
# Cache lokal agar tidak perlu menulis ke folder instalasi Python.
os.environ.setdefault('NUMBA_CACHE_DIR', str(Path(__file__).resolve().parents[1] / 'data' / 'numba_cache'))
import numpy as np
import librosa
import soundfile as sf
import pandas as pd


def file_hash(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def inspect_audio(path):
    info = sf.info(path)
    # Baca seluruh rekaman bertahap untuk menemukan error decode dan audio kosong.
    peak = 0.0
    with sf.SoundFile(path) as audio:
        for block in audio.blocks(blocksize=65536):
            if not np.isfinite(block).all():
                raise ValueError('Audio berisi nilai tidak valid')
            peak = max(peak, float(np.max(np.abs(block))))
    if info.frames == 0 or peak == 0:
        raise ValueError('Audio kosong atau seluruhnya hening')
    return dict(duration=info.duration, sample_rate=info.samplerate,
                channels=info.channels, sha256=file_hash(path))


def load_audio(path, config):
    if not Path(path).is_file():
        raise ValueError(f'File tidak ditemukan: {path}')
    try:
        # Satu rekaman satu sampel. Batasi durasi agar RAM dan waktu terkendali.
        y, sr = librosa.load(path, sr=config['sample_rate'], mono=True,
                             duration=config['max_duration'])
    except Exception as e:
        raise ValueError(f'Audio tidak dapat dibaca: {path}. {e}') from e
    if len(y) < 2048 or not np.isfinite(y).all() or np.max(np.abs(y)) == 0:
        raise ValueError(f'Audio terlalu pendek, kosong, atau tidak valid: {path}')
    # Tidak melakukan normalisasi peak: perbedaan energi tetap bisa dianalisis.
    return y, sr


def extract_features(path, config):
    y, sr = load_audio(path, config)
    spectrum = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))
    mfcc = librosa.feature.mfcc(S=librosa.power_to_db(
        librosa.feature.melspectrogram(S=spectrum ** 2, sr=sr)), n_mfcc=20)
    values = {
        'rms': librosa.feature.rms(S=spectrum),
        'zcr': librosa.feature.zero_crossing_rate(y),
        'centroid': librosa.feature.spectral_centroid(S=spectrum, sr=sr),
        'bandwidth': librosa.feature.spectral_bandwidth(S=spectrum, sr=sr),
        'rolloff': librosa.feature.spectral_rolloff(S=spectrum, sr=sr),
        'flatness': librosa.feature.spectral_flatness(S=spectrum),
        'contrast': librosa.feature.spectral_contrast(S=spectrum, sr=sr),
        'mfcc': mfcc,
        'delta': librosa.feature.delta(mfcc, mode='nearest'),
        'delta2': librosa.feature.delta(mfcc, order=2, mode='nearest'),
    }
    result = {}
    for name, array in values.items():
        for i, row in enumerate(array):
            result[f'{name}_{i+1}_mean'] = float(np.mean(row))
            result[f'{name}_{i+1}_std'] = float(np.std(row))
    if not np.isfinite(list(result.values())).all():
        raise ValueError(f'Fitur tidak valid: {path}')
    return result


def predict_bird_species(audio_path, top_k=3, model_path='models/best_model.joblib'):
    import joblib
    bundle = joblib.load(model_path)
    features = extract_features(audio_path, bundle['config'])
    X = pd.DataFrame([features])[bundle['features']]
    model, encoder = bundle['model'], bundle['encoder']
    prediction = encoder.inverse_transform(model.predict(X).astype(int))[0]
    result = {'prediction': prediction, 'probabilities_calibrated': False}
    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(X)[0]
        order = np.argsort(probabilities)[::-1][:max(1, min(top_k, len(probabilities)))]
        result['top_k'] = [{'species': encoder.inverse_transform([int(model.classes_[i])])[0],
                            'probability': float(probabilities[i])} for i in order]
    else:
        result['note'] = 'Model ini tidak menyediakan probabilitas.'
    return result
