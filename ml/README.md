# `ml/` - Sheet Sage 2 ML package

End-to-end audio-to-lead-sheet pipeline, owned by Agent C.

## Pipeline order

```
audio file/URL
    -> ml.audio_io.load_audio
    -> ml.models.demucs.DemucsWrapper            (stems; PR-1)
    -> ml.models.beat_transformer.BeatTransformerWrapper  (beats; PR-2)
    -> ml.models.mert.MERTFeatureExtractor       (features; PR-3)
        -> melody head            (PR-4/5)  -> list[Note]
        -> chord head + refine    (PR-6/7)  -> list[Chord]
        -> key head               (PR-8)    -> Key
        -> voicing model          (PR-9/10) -> list[Voicing]
    -> quantization to beat grid                 (PR-11)
    -> ml.scoring.musicxml.build_musicxml        (PR-12)
        -> ml.scoring.lilypond.build_lilypond_pdf
        -> ml.scoring.midi.build_midi
    => ml.types.TranscriptionResult
```

## Where things live

| Concern | Module |
|---|---|
| Shared dataclasses (Beat, Note, Chord, Key, Voicing, TranscriptionResult) | `ml.types` |
| Audio loading + YouTube + hashing | `ml.audio_io` |
| Pretrained model wrappers | `ml.models.{demucs,beat_transformer,mert}` |
| Model registry (hashes, licenses, eval metrics) | `ml/models/registry.yaml` |
| Dataset loaders (HookTheory, POP909, Isophonics, Billboard, RWC-Pop) | `ml.datasets.*` |
| Training loops + compute notes | `ml/training/` |
| End-to-end pipeline (Agent B's contract) | `ml.inference.transcribe` |
| Score writers (canonical = MusicXML) | `ml.scoring.{musicxml,lilypond,midi}` |

## Links

- Team contract: [`../skills.md`](../skills.md)
- PR DAG: [`../.orchestrator/dependency_graph.yaml`](../.orchestrator/dependency_graph.yaml)
- Training compute target + kill criterion: [`training/README.md`](training/README.md)
