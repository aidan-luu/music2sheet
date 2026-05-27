"""Beat + downbeat tracking wrapper.

This PR (PR-2) ships the ``madmom`` backend only. The Beat Transformer
backend (Zhao et al., 2022; arXiv:2209.07140) is gated behind a
``NotImplementedError`` because the official implementation at
https://github.com/zhaojw1998/Beat-Transformer is not pip-installable and a
local re-implementation would balloon this PR's scope. It is filed as a
follow-up; see ``# TODO(PR-2-followup)`` below.

The public surface is a single class:

* :class:`BeatTransformerWrapper` — selectable backend, single ``detect``
  entry point that returns a sorted ``list[Beat]``.

The downbeat flag on each :class:`~ml.types.Beat` is derived by aligning the
beat track against an independent downbeat track within a ``DOWNBEAT_TOL_S``
tolerance window. This avoids relying on either model's internal bar
inference, which historically disagree on the first downbeat for songs
with anacruses.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from ml.types import Beat

if TYPE_CHECKING:
    # Imported lazily inside methods to keep import-time cost low and to
    # avoid hard-failing the module import when madmom is unavailable in
    # a given environment (e.g. on Python 3.11+ where madmom's wheels lag).
    pass


_SUPPORTED_BACKENDS = ("madmom", "beat_transformer")

#: Maximum gap (seconds) between a beat and a downbeat candidate for the
#: beat to be flagged ``downbeat=True``. 50 ms matches the mir_eval beat F1
#: tolerance commonly used in the literature.
DOWNBEAT_TOL_S = 0.050


class BeatTransformerWrapper:
    """Beat + downbeat detector with selectable backend.

    Parameters
    ----------
    backend:
        Either ``"madmom"`` (default, implemented here) or
        ``"beat_transformer"`` (raises :class:`NotImplementedError` in this
        PR; see TODO below).
    device:
        Reserved for the ``beat_transformer`` backend. Ignored by
        ``madmom``, which runs on CPU.

    Raises
    ------
    ValueError
        If ``backend`` is not one of the supported names.
    """

    def __init__(self, backend: str = "madmom", device: str | None = None) -> None:
        if backend not in _SUPPORTED_BACKENDS:
            raise ValueError(
                f"Unknown backend {backend!r}; expected one of {_SUPPORTED_BACKENDS}."
            )
        self.backend = backend
        self.device = device

    # ------------------------------------------------------------------ API

    def detect(
        self,
        audio_path: str | Path,
        stems: dict[str, np.ndarray] | None = None,
    ) -> list[Beat]:
        """Detect beats and downbeats in an audio file.

        Parameters
        ----------
        audio_path:
            Path to the source mix. Used as the fallback madmom input when
            ``stems`` is ``None``.
        stems:
            Optional output of :class:`~ml.models.demucs.DemucsWrapper`. If
            provided and a ``"drums"`` entry is present, the drums stem is
            mixed to mono and fed to madmom in place of the full mix. This
            consistently improves downbeat accuracy on full-band pop/rock.

        Returns
        -------
        list[Beat]
            Time-sorted list of :class:`~ml.types.Beat` events. The
            ``downbeat`` flag is set when the beat aligns with the
            independent downbeat track within ``DOWNBEAT_TOL_S``.

        Raises
        ------
        NotImplementedError
            If the wrapper was constructed with ``backend="beat_transformer"``.
        """
        if self.backend == "beat_transformer":
            # TODO(PR-2-followup): port the Beat Transformer inference path
            # from https://github.com/zhaojw1998/Beat-Transformer (Zhao et
            # al., 2022, arXiv:2209.07140). The official repo is not
            # pip-installable; either vendor the minimal inference modules
            # under ml/models/_beat_transformer_src/ or wait for an
            # upstream PyPI release.
            raise NotImplementedError(
                "Beat Transformer integration deferred; see PR-2 followup "
                "(arXiv:2209.07140, https://github.com/zhaojw1998/Beat-Transformer)."
            )

        return self._detect_madmom(audio_path, stems)

    # --------------------------------------------------------------- madmom

    def _detect_madmom(
        self,
        audio_path: str | Path,
        stems: dict[str, np.ndarray] | None,
    ) -> list[Beat]:
        """Run the madmom beat + downbeat pipeline and merge the two tracks."""
        # Imported lazily so that ``backend="beat_transformer"`` users can
        # construct the wrapper on a machine without madmom installed.
        from madmom.features.beats import (  # noqa: PLC0415
            DBNBeatTrackingProcessor,
            RNNBeatProcessor,
        )
        from madmom.features.downbeats import (  # noqa: PLC0415
            DBNDownBeatTrackingProcessor,
            RNNDownBeatProcessor,
        )

        beat_input = self._prepare_input(audio_path, stems)

        # --- beats -----------------------------------------------------
        # 55-215 BPM brackets the bulk of pop/rock/jazz; matches the
        # default search range in madmom's docs.
        beat_act = RNNBeatProcessor()(beat_input)
        beat_tracker = DBNBeatTrackingProcessor(
            min_bpm=55.0, max_bpm=215.0, fps=100, transition_lambda=100
        )
        beat_times = np.asarray(beat_tracker(beat_act), dtype=np.float64)

        # --- downbeats -------------------------------------------------
        # beats_per_bar=[3, 4] covers 3/4 and 4/4; the DBN picks per-song.
        db_act = RNNDownBeatProcessor()(beat_input)
        db_tracker = DBNDownBeatTrackingProcessor(
            beats_per_bar=[3, 4], fps=100
        )
        # Downbeat tracker returns (time, position_in_bar) pairs; we keep
        # the times whose position == 1 (i.e. bar starts).
        db_pairs = np.asarray(db_tracker(db_act), dtype=np.float64)
        if db_pairs.ndim == 2 and db_pairs.shape[1] >= 2:
            downbeat_times = db_pairs[db_pairs[:, 1] == 1.0, 0]
        else:  # defensive: empty / unexpected shape
            downbeat_times = np.empty(0, dtype=np.float64)

        # --- merge -----------------------------------------------------
        return _merge_beats_downbeats(beat_times, downbeat_times)

    # --------------------------------------------------------------- utils

    @staticmethod
    def _prepare_input(
        audio_path: str | Path,
        stems: dict[str, np.ndarray] | None,
    ) -> str | np.ndarray:
        """Pick what to feed madmom: drums stem if given, else the file path.

        Madmom's processors accept either a file path (str) or an in-memory
        signal (numpy array). Returning a path lets madmom handle decoding.
        """
        if stems is not None and "drums" in stems:
            drums = np.asarray(stems["drums"])
            if drums.ndim == 2:
                # (channels, samples) -> mono mix-down.
                drums = drums.mean(axis=0)
            return drums.astype(np.float32, copy=False)
        return str(audio_path)


def _merge_beats_downbeats(
    beat_times: np.ndarray,
    downbeat_times: np.ndarray,
    tol: float = DOWNBEAT_TOL_S,
) -> list[Beat]:
    """Flag each beat as a downbeat iff a downbeat falls within ``tol`` seconds.

    Parameters
    ----------
    beat_times, downbeat_times:
        1-D arrays of timestamps in seconds. May be empty.
    tol:
        Half-window for the nearest-neighbour match. Defaults to
        :data:`DOWNBEAT_TOL_S`.

    Returns
    -------
    list[Beat]
        Sorted by ``time``. ``confidence`` is set to a constant 1.0 because
        madmom's DBN trackers emit hard decisions (no per-beat score).
    """
    beat_times = np.sort(np.asarray(beat_times, dtype=np.float64))
    downbeat_times = np.sort(np.asarray(downbeat_times, dtype=np.float64))

    out: list[Beat] = []
    for t in beat_times:
        is_downbeat = False
        if downbeat_times.size > 0:
            # np.searchsorted gives O(log n) nearest-neighbour lookup.
            idx = int(np.searchsorted(downbeat_times, t))
            candidates = []
            if idx < downbeat_times.size:
                candidates.append(downbeat_times[idx])
            if idx > 0:
                candidates.append(downbeat_times[idx - 1])
            if any(abs(float(c) - float(t)) <= tol for c in candidates):
                is_downbeat = True
        out.append(Beat(time=float(t), downbeat=is_downbeat, confidence=1.0))
    return out
