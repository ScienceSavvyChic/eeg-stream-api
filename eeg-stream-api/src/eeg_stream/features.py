from __future__ import annotations

import numpy as np
from scipy.signal import coherence, welch

BANDS = {
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
}


def band_power(data_uv: np.ndarray, sfreq: float) -> list[dict]:
    """Per-channel band power (µV²) from shape (n_channels, n_samples)."""
    rows: list[dict] = []
    for ch_idx in range(data_uv.shape[0]):
        freqs, psd = welch(data_uv[ch_idx], fs=sfreq, nperseg=min(256, data_uv.shape[1]))
        for band, (fmin, fmax) in BANDS.items():
            mask = (freqs >= fmin) & (freqs <= fmax)
            power = float(np.trapz(psd[mask], freqs[mask])) if mask.any() else 0.0
            rows.append({"channel_index": ch_idx, "band": band, "power_uv2": power})
    return rows


def mean_alpha_beta_ratio(data_uv: np.ndarray, sfreq: float) -> float:
    """Simple cognitive-performance proxy: mean alpha / beta across channels."""
    alpha_vals: list[float] = []
    beta_vals: list[float] = []
    for ch_idx in range(data_uv.shape[0]):
        freqs, psd = welch(data_uv[ch_idx], fs=sfreq, nperseg=min(256, data_uv.shape[1]))
        a = (freqs >= 8) & (freqs <= 13)
        b = (freqs >= 13) & (freqs <= 30)
        alpha_vals.append(float(np.trapz(psd[a], freqs[a])) if a.any() else 1e-12)
        beta_vals.append(float(np.trapz(psd[b], freqs[b])) if b.any() else 1e-12)
    return float(np.mean(alpha_vals) / np.mean(beta_vals))


def pairwise_coherence(
    data_uv: np.ndarray,
    sfreq: float,
    fmin: float = 4.0,
    fmax: float = 30.0,
) -> dict:
    """Average magnitude coherence between channel pairs in [fmin, fmax]."""
    n_ch = data_uv.shape[0]
    if n_ch < 2:
        return {"fmin_hz": fmin, "fmax_hz": fmax, "pairs": []}

    pairs: list[dict] = []
    for i in range(n_ch):
        for j in range(i + 1, n_ch):
            freqs, coh = coherence(
                data_uv[i],
                data_uv[j],
                fs=sfreq,
                nperseg=min(256, data_uv.shape[1]),
            )
            mask = (freqs >= fmin) & (freqs <= fmax)
            val = float(np.mean(coh[mask])) if mask.any() else 0.0
            pairs.append({"i": i, "j": j, "coherence": round(val, 4)})
    return {"fmin_hz": fmin, "fmax_hz": fmax, "pairs": pairs}
