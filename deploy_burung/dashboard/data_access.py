"""Cari audio lengkap atau contoh kecil pada paket deploy."""
from pathlib import Path
import pandas as pd


def resolve_audio(root, relative):
    root = Path(root).resolve()
    path = (root / str(relative).replace("\\", "/")).resolve()
    dataset = (root / "indonesian-bird-sounds").resolve()
    if not path.is_relative_to(dataset):
        raise ValueError("Lokasi rekaman berada di luar dataset.")
    if path.is_file():
        return path

    manifest = root / "sample_audio/manifest.csv"
    if manifest.is_file():
        samples = pd.read_csv(manifest)
        # Metadata lama menggunakan pemisah Windows, sedangkan server memakai Linux.
        key = str(relative).replace("\\", "/")
        matches = samples[samples["path"].str.replace("\\", "/", regex=False) == key]
        if not matches.empty:
            sample = (root / matches.iloc[0]["sample_path"]).resolve()
            if not sample.is_relative_to((root / "sample_audio").resolve()):
                raise ValueError("Lokasi contoh audio tidak valid.")
            if sample.is_file():
                return sample
    return None


def available_paths(root, paths):
    # Baca manifest sekali, lalu periksa file yang benar-benar tersedia.
    root = Path(root).resolve()
    manifest = root / "sample_audio/manifest.csv"
    samples = pd.read_csv(manifest) if manifest.is_file() else pd.DataFrame()
    available = set()
    if not samples.empty:
        for row in samples.itertuples():
            sample = (root / row.sample_path).resolve()
            if sample.is_relative_to((root / "sample_audio").resolve()) and sample.is_file():
                available.add(str(row.path).replace("\\", "/"))
    for relative in paths:
        key = str(relative).replace("\\", "/")
        path = (root / key).resolve()
        if path.is_relative_to((root / "indonesian-bird-sounds").resolve()) and path.is_file():
            available.add(key)
    return [p for p in paths if str(p).replace("\\", "/") in available]
