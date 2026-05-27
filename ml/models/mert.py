"""MERT-v1-330M feature-extractor wrapper.

Wraps the Hugging Face checkpoint ``m-a-p/MERT-v1-330M`` (Apache-2.0) and
exposes a small, typed API that downstream heads (PR-4..PR-7) can pin
against. The standard recipe used by the SheetSage / chord / melody heads
is "sum the last 4 hidden states" of the transformer, which empirically
beats taking only the final hidden state (Chen et al., 2023).

Frame rate of the MERT encoder is 75 Hz at its native 24 kHz input SR.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing only
    import torch


# MERT-v1-330M expects mono audio at this sample rate.
MERT_NATIVE_SR: int = 24000
# Native frame rate of the MERT encoder output (hidden states per second).
MERT_FRAME_RATE_HZ: int = 75
# Chunk length (seconds) for naive long-audio splitting. Anything longer
# than this is split into back-to-back, non-overlapping windows before
# being fed to the model. See the comment in ``extract`` for caveats.
MERT_CHUNK_SECONDS: int = 30


def _autodetect_device() -> str:
    """Return the best available torch device string: cuda > mps > cpu."""
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _file_sha256_prefix(path: Path, prefix_len: int = 16) -> str:
    """Hex SHA-256 of the file bytes, truncated to ``prefix_len`` chars.

    Fallback hasher used only if :func:`ml.audio_io.audio_hash` is not yet
    implemented (PR-1 owns it). Remove this helper once PR-1 lands and
    audio_hash is callable.
    """
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:prefix_len]


def _hash_audio_path(path: Path) -> str:
    """Stable cache key for an audio file.

    Tries :func:`ml.audio_io.audio_hash` first (owned by PR-1); falls back
    to a local SHA-256 prefix if PR-1 has not landed yet. The fallback
    branch will be removed once PR-1 merges.
    """
    try:
        from ml.audio_io import audio_hash

        return audio_hash(path)
    except NotImplementedError:
        # TODO(PR-1 follow-up): drop this fallback once audio_hash ships.
        return _file_sha256_prefix(path)


def _resample(waveform: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample a 1-D float32 waveform to ``target_sr``. No-op when SRs match."""
    if orig_sr == target_sr:
        return waveform.astype(np.float32, copy=False)
    import torch
    import torchaudio.functional as AF

    tensor = torch.from_numpy(np.ascontiguousarray(waveform, dtype=np.float32))
    resampled = AF.resample(tensor, orig_freq=orig_sr, new_freq=target_sr)
    return resampled.cpu().numpy().astype(np.float32, copy=False)


def _to_mono(waveform: np.ndarray) -> np.ndarray:
    """Collapse a multi-channel waveform to mono by averaging channels."""
    if waveform.ndim == 1:
        return waveform.astype(np.float32, copy=False)
    if waveform.ndim == 2:
        # Assume (channels, samples) if the first axis is the smaller one,
        # otherwise (samples, channels).
        if waveform.shape[0] <= waveform.shape[1]:
            return waveform.mean(axis=0).astype(np.float32, copy=False)
        return waveform.mean(axis=1).astype(np.float32, copy=False)
    raise ValueError(f"Unsupported waveform shape {waveform.shape!r}; expected 1-D or 2-D")


class MERTFeatureExtractor:
    """Hugging Face ``m-a-p/MERT-v1-330M`` feature extractor wrapper.

    The model + processor are loaded lazily on the first call to
    :meth:`extract` so that import-time cost stays near zero and CI does
    not pay the download tax unless a slow test is selected.
    """

    def __init__(
        self,
        model_id: str = "m-a-p/MERT-v1-330M",
        device: str | None = None,
        cache_dir: Path | None = None,
        layers_to_sum: int = 4,
    ) -> None:
        if layers_to_sum < 1:
            raise ValueError(f"layers_to_sum must be >= 1, got {layers_to_sum}")

        self.model_id: str = model_id
        self.device: str = device if device is not None else _autodetect_device()
        self.cache_dir: Path | None = Path(cache_dir) if cache_dir is not None else None
        self.layers_to_sum: int = layers_to_sum

        # Lazy state — populated on first extract() call.
        self._model: Any | None = None
        self._processor: Any | None = None
        self._torch_device: torch.device | None = None

    # ------------------------------------------------------------------ #
    # Lazy loader
    # ------------------------------------------------------------------ #
    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return

        import torch
        from transformers import AutoFeatureExtractor, AutoModel

        # MERT ships custom Python modules in its repo for the conv frontend,
        # so trust_remote_code is required for both the model and processor.
        self._processor = AutoFeatureExtractor.from_pretrained(
            self.model_id, trust_remote_code=True
        )
        model = AutoModel.from_pretrained(self.model_id, trust_remote_code=True)
        self._torch_device = torch.device(self.device)
        model = model.to(self._torch_device)
        model.eval()
        self._model = model

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def extract(
        self,
        audio: np.ndarray | str | Path,
        sr: int = 44100,
    ) -> np.ndarray:
        """Extract MERT features.

        Parameters
        ----------
        audio:
            Either a numpy waveform (1-D mono or 2-D multichannel) or a
            filesystem path to an audio file readable by ``librosa``.
        sr:
            Sample rate of ``audio`` when it is a numpy array. Ignored if
            ``audio`` is a path (the file's own SR is used). The waveform
            is resampled to MERT's native 24 kHz internally.

        Returns
        -------
        np.ndarray
            Shape ``(T, D)`` float32 array, with ``T`` at 75 Hz and ``D``
            equal to MERT's hidden size (1024 for v1-330M). Computed as
            the sum of the last ``layers_to_sum`` transformer hidden
            states, following the standard MERT downstream recipe.
        """
        import torch

        # Resolve to a mono numpy waveform at MERT's native SR.
        if isinstance(audio, (str, Path)):
            import librosa

            waveform_np, file_sr = librosa.load(
                str(audio), sr=MERT_NATIVE_SR, mono=True
            )
            waveform_np = waveform_np.astype(np.float32, copy=False)
            effective_sr = file_sr  # already MERT_NATIVE_SR from librosa
        else:
            waveform_np = _to_mono(np.asarray(audio))
            waveform_np = _resample(waveform_np, sr, MERT_NATIVE_SR)
            effective_sr = MERT_NATIVE_SR

        assert effective_sr == MERT_NATIVE_SR, "internal: resample target mismatch"

        self._ensure_loaded()
        assert self._model is not None and self._processor is not None
        assert self._torch_device is not None

        # Naive chunking: split into back-to-back 30 s windows. This is a
        # placeholder — the streaming-optimal version overlaps windows and
        # discards a context margin at the chunk boundaries to avoid the
        # attention edge artifacts. Fine for v0 transcription accuracy.
        chunk_samples = MERT_CHUNK_SECONDS * MERT_NATIVE_SR
        if waveform_np.shape[0] <= chunk_samples:
            chunks: list[np.ndarray] = [waveform_np]
        else:
            chunks = [
                waveform_np[start : start + chunk_samples]
                for start in range(0, waveform_np.shape[0], chunk_samples)
            ]

        feature_blocks: list[np.ndarray] = []
        with torch.no_grad():
            for chunk in chunks:
                inputs = self._processor(
                    chunk,
                    sampling_rate=MERT_NATIVE_SR,
                    return_tensors="pt",
                )
                inputs = {k: v.to(self._torch_device) for k, v in inputs.items()}
                outputs = self._model(**inputs, output_hidden_states=True)

                # hidden_states is a tuple of length (num_layers + 1) where
                # index 0 is the conv-frontend embedding. Sum the last
                # ``layers_to_sum`` transformer layers.
                hidden_states = outputs.hidden_states
                selected = hidden_states[-self.layers_to_sum :]
                stacked = torch.stack(selected, dim=0)  # (L, B, T, D)
                summed = stacked.sum(dim=0)  # (B, T, D)
                feature_blocks.append(summed.squeeze(0).cpu().float().numpy())

        return np.concatenate(feature_blocks, axis=0).astype(np.float32, copy=False)

    def extract_cached(self, audio_path: str | Path) -> Path:
        """Extract features for ``audio_path``, caching to ``cache_dir``.

        The cache filename is ``<hash>.mert.npy`` where ``<hash>`` comes
        from :func:`ml.audio_io.audio_hash`. If the cache file already
        exists it is returned without re-running the model.
        """
        if self.cache_dir is None:
            raise ValueError(
                "MERTFeatureExtractor.cache_dir must be set to use extract_cached()"
            )

        audio_path = Path(audio_path)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        digest = _hash_audio_path(audio_path)
        cache_path = self.cache_dir / f"{digest}.mert.npy"
        if cache_path.exists():
            return cache_path

        features = self.extract(audio_path)
        # Atomic-ish write: stage to a sibling .tmp then rename.
        tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
        np.save(tmp_path, features)
        tmp_path.replace(cache_path)
        return cache_path
