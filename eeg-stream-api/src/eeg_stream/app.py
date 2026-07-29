from __future__ import annotations

import asyncio

import numpy as np
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

from eeg_stream.features import band_power, mean_alpha_beta_ratio, pairwise_coherence
from eeg_stream.synthetic import MockPiEEGSource, generate_window

app = FastAPI(
    title="EEG Stream API",
    description="Windowed EEG features and mock PiEEG-style streaming for neurotech prototypes.",
    version="0.1.0",
)

_mock = MockPiEEGSource()


class EEGWindow(BaseModel):
    sfreq_hz: float = Field(gt=0, description="Sampling rate in Hz")
    data_uv: list[list[float]] = Field(description="2D list: channels × samples (microvolts)")
    channel_names: list[str] | None = None


class FeatureResponse(BaseModel):
    band_power: list[dict]
    alpha_beta_ratio: float
    focus_index: float


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": "eeg-stream-api"}


@app.get("/v1/demo/window")
def demo_window(channels: int = 8, samples: int = 500, sfreq: float = 250.0) -> dict:
    data, names, rate = generate_window(n_channels=channels, n_samples=samples, sfreq=sfreq, seed=7)
    return {"channel_names": names, "sfreq_hz": rate, "data_uv": data.tolist()}


@app.get("/v1/mock/pieeg/next")
def mock_pieeg_next() -> dict:
    """Next window from an in-memory mock edge buffer (PiEEG-style)."""
    return _mock.next_window()


@app.post("/v1/features", response_model=FeatureResponse)
def compute_features(window: EEGWindow) -> FeatureResponse:
    arr = np.asarray(window.data_uv, dtype=float)
    if arr.ndim != 2:
        raise HTTPException(400, "data_uv must be 2D: channels × samples")
    if arr.shape[1] < 64:
        raise HTTPException(400, "need at least 64 samples per channel")

    bp = band_power(arr, window.sfreq_hz)
    ratio = mean_alpha_beta_ratio(arr, window.sfreq_hz)
    # Normalized stub index for demos (not a clinical metric)
    focus = float(np.clip(ratio / (ratio + 1.0), 0.0, 1.0))
    return FeatureResponse(band_power=bp, alpha_beta_ratio=round(ratio, 4), focus_index=round(focus, 4))


@app.post("/v1/connectivity/coherence")
def connectivity(window: EEGWindow, fmin: float = 4.0, fmax: float = 30.0) -> dict:
    arr = np.asarray(window.data_uv, dtype=float)
    if arr.ndim != 2:
        raise HTTPException(400, "data_uv must be 2D")
    return pairwise_coherence(arr, window.sfreq_hz, fmin=fmin, fmax=fmax)


@app.websocket("/v1/ws/stream")
async def ws_stream(websocket: WebSocket) -> None:
    """Push mock EEG windows once per second (edge prototyping)."""
    await websocket.accept()
    source = MockPiEEGSource()
    try:
        while True:
            await websocket.send_json(source.next_window())
            await asyncio.sleep(1.0)
    except WebSocketDisconnect:
        return
