"""HT-Demucs v4 source-separation wrapper.

Wraps the official `demucs` Python package (Meta AI). HT-Demucs v4 separates
an input mix into ``vocals``, ``drums``, ``bass``, and ``other`` stems at
44.1 kHz. Downstream:
* the chord head (PR-6/7) consumes the ``no-drums`` mix (vocals + bass + other),
* the melody head (PR-4/5) consumes the ``vocals`` stem,
* the beat tracker (PR-2) consumes the full mix and may use ``drums``.

Design notes
------------
* The wrapped model is lazy-loaded on first call to :meth:`DemucsWrapper.separate`
  so module import stays cheap (CI doesn't pull the ~80 MB checkpoint).
* Device autodetect order is CUDA -> MPS -> CPU. Callers may override with
  the ``device`` kwarg.
* :meth:`DemucsWrapper.separate_cached` keys results on
  :func:`ml.audio_io.audio_hash` and short-circuits on a cache hit. Cached
  stems are written as float32 WAVs at 44.1 kHz.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from ml.audio_io import audio_hash

if TYPE_CHECKING:  # pragma: no cover - import only used for type hints
    import torch

# HT-Demucs v4 expects 44.1 kHz stereo input and produces output at the same
# sample rate. Hardcoded because the model architecture is fixed.
DEMUCS_SR: int = 44100

# Standard 4-stem layout for htdemucs / htdemucs_ft. We re-export this so
# downstream code can iterate without importing demucs directly.
STEM_NAMES: tuple[str, ...] = ("drums", "bass", "other", "vocals")


def _autodetect_device() -> str:
    """Pick the best available torch device. CUDA > MPS > CPU."""
    try:
        import torch
    except ImportError:
        # No torch installed (e.g. metadata-only test env). Caller should
        # never actually invoke separate() in that scenario, but returning
        # 'cpu' keeps construction harmless.
        return "cpu"

    if torch.cuda.is_available():
        return "cuda"
    # MPS is the Apple Silicon backend; .is_available() exists from torch>=1.12.
    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        return "mps"
    return "cpu"


class DemucsWrapper:
    """Thin wrapper over the official ``demucs`` package's HT-Demucs v4 model.

    Parameters
    ----------
    model_name:
        Name of the demucs bag/model to load. ``"htdemucs_ft"`` is the
        fine-tuned 4-stem v4 variant (best quality, slower). ``"htdemucs"``
        is the base v4 model.
    device:
        Torch device string. ``None`` triggers autodetect (CUDA > MPS > CPU).
    cache_dir:
        Directory under which :meth:`separate_cached` writes per-input
        stem folders. Defaults to ``./.cache/demucs`` relative to cwd.
    """

    def __init__(
        self,
        model_name: str = "htdemucs_ft",
        device: str | None = None,
        cache_dir: Path | None = None,
    ) -> None:
        self.model_name: str = model_name
        self.device: str = device if device is not None else _autodetect_device()
        self.cache_dir: Path = (
            Path(cache_dir) if cache_dir is not None else Path(".cache") / "demucs"
        )
        # Lazy-loaded fields. The model is fetched + initialised on first
        # call to separate() so simply constructing a wrapper is cheap.
        self._model: Any | None = None
        self._sample_rate: int = DEMUCS_SR

    # ------------------------------------------------------------------ #
    # Model lifecycle
    # ------------------------------------------------------------------ #
    def _ensure_model_loaded(self) -> Any:
        """Load (or return cached) demucs model on the configured device."""
        if self._model is not None:
            return self._model

        # Local imports keep `import ml.models.demucs` cheap.
        import torch
        from demucs.pretrained import get_model

        model = get_model(name=self.model_name)
        model.to(torch.device(self.device))
        model.eval()
        self._model = model
        # demucs models expose their training sample rate; fall back to 44.1k.
        self._sample_rate = int(getattr(model, "samplerate", DEMUCS_SR))
        return model

    # ------------------------------------------------------------------ #
    # Inference
    # ------------------------------------------------------------------ #
    def separate(self, audio_path: str | Path) -> dict[str, np.ndarray]:
        """Separate the input mix into stems.

        Parameters
        ----------
        audio_path:
            Path to a local audio file (any format readable by demucs's
            internal loader, which uses torchaudio + ffmpeg).

        Returns
        -------
        dict[str, np.ndarray]
            ``{"vocals", "drums", "bass", "other"}`` mapping to float32
            arrays shaped ``(channels, samples)`` at 44.1 kHz.
        """
        import torch
        from demucs.apply import apply_model
        from demucs.audio import convert_audio

        model = self._ensure_model_loaded()
        device = torch.device(self.device)
        model_sr = int(getattr(model, "samplerate", DEMUCS_SR))
        model_channels = int(getattr(model, "audio_channels", 2))

        # _load_audio_for_demucs returns a stereo tensor at DEMUCS_SR (44.1k).
        # We still route through convert_audio to defensively match whatever
        # samplerate/channel count the loaded model declares.
        wav = _load_audio_for_demucs(Path(audio_path))
        wav = convert_audio(
            wav,
            from_samplerate=DEMUCS_SR,
            to_samplerate=model_sr,
            channels=model_channels,
        )

        # demucs apply_model expects shape (batch, channels, samples).
        with torch.no_grad():
            batched = wav.unsqueeze(0).to(device)
            # progress=False keeps CI logs clean. split=True chunks long
            # inputs and is the default for HT-Demucs.
            out = apply_model(model, batched, device=device, progress=False)
        # out shape: (batch=1, sources, channels, samples)
        out_np = out[0].cpu().numpy().astype(np.float32)

        # demucs exposes the source order on the model as `sources`.
        sources: list[str] = list(getattr(model, "sources", STEM_NAMES))
        return {name: out_np[i] for i, name in enumerate(sources)}

    def separate_cached(self, audio_path: str | Path) -> dict[str, Path]:
        """Same as :meth:`separate` but persists stems to disk.

        Stems are written to ``self.cache_dir / "<hash>" / "{stem}.wav"`` and
        the call short-circuits on cache hit. The cache key is the SHA-256
        digest of the input file bytes (truncated; see
        :func:`ml.audio_io.audio_hash`).

        Returns
        -------
        dict[str, Path]
            Mapping of stem name to on-disk WAV path.
        """
        audio_path = Path(audio_path)
        key = audio_hash(audio_path)
        out_dir = self.cache_dir / key
        expected = {name: out_dir / f"{name}.wav" for name in STEM_NAMES}

        if out_dir.is_dir() and all(p.exists() for p in expected.values()):
            return expected

        stems = self.separate(audio_path)
        out_dir.mkdir(parents=True, exist_ok=True)

        # Local import to avoid a hard soundfile dep at module import time.
        import soundfile as sf

        paths: dict[str, Path] = {}
        for name, arr in stems.items():
            target = out_dir / f"{name}.wav"
            # soundfile expects (samples, channels) for multi-channel.
            data = arr.T if arr.ndim == 2 else arr
            sf.write(str(target), data, self._sample_rate, subtype="FLOAT")
            paths[name] = target
        return paths


def _load_audio_for_demucs(path: Path) -> "torch.Tensor":
    """Load audio as a float32 torch tensor shaped (channels, samples).

    Prefers ``demucs.audio.AudioFile`` (which uses ffmpeg + torchaudio under
    the hood and handles any format demucs natively supports) and falls back
    to torchaudio's direct loader for the common WAV/FLAC case.
    """
    import torch

    try:
        from demucs.audio import AudioFile

        wav = AudioFile(str(path)).read(streams=0, samplerate=DEMUCS_SR, channels=2)
        return wav.to(torch.float32)
    except Exception:
        import torchaudio

        wav, sr = torchaudio.load(str(path))
        if sr != DEMUCS_SR:
            wav = torchaudio.functional.resample(wav, sr, DEMUCS_SR)
        if wav.shape[0] == 1:
            # mono -> stereo by duplication
            wav = wav.repeat(2, 1)
        elif wav.shape[0] > 2:
            wav = wav[:2]
        return wav.to(torch.float32)
