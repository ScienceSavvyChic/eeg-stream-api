from __future__ import annotations

import numpy as np

DEFAULT_CHANNELS = ("Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2")


def generate_window(
    *,
    n_channels: int = 8,
    n_samples: int = 500,
    sfreq: float = 250.0,
    seed: int | None = None,
) -> tuple[np.ndarray, list[str], float]:
    """Return (channels × samples) µV-scale synthetic EEG window."""
    rng = np.random.default_rng(seed)
    names = list(DEFAULT_CHANNELS[:n_channels])
    if len(names) < n_channels:
        names.extend([f"E{i + 1}" for i in range(len(names), n_channels)])

    t = np.arange(n_samples) / sfreq
    data = rng.normal(0, 10.0, size=(n_channels, n_samples))
    alpha = np.sin(2 * np.pi * 10.0 * t) * 25.0
    for i in range(n_channels):
        data[i] += alpha * (0.5 + 0.5 * rng.random())
    return data, names, sfreq


class MockPiEEGSource:
    """Yields fixed-size windows like an edge acquisition buffer."""

    def __init__(
        self,
        sfreq: float = 250.0,
        window_sec: float = 2.0,
        n_channels: int = 8,
        seed: int = 42,
    ) -> None:
        self.sfreq = sfreq
        self.n_samples = int(window_sec * sfreq)
        self.n_channels = n_channels
        self._seed = seed
        self._tick = 0

    def next_window(self) -> dict:
        data, names, sfreq = generate_window(
            n_channels=self.n_channels,
            n_samples=self.n_samples,
            sfreq=self.sfreq,
            seed=self._seed + self._tick,
        )
        self._tick += 1
        return {
            "channel_names": names,
            "sfreq_hz": sfreq,
            "data_uv": data.tolist(),
            "sequence": self._tick,
        }
