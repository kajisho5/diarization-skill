# Changelog

## 0.1.0 - 2026-10-30

First release (roadmap steps 2-4).

- `diarization diarize`: media -> speaker turns (`diarization-skill/turns@1`, `schemas/turns.schema.json`). pyannote.audio 4.x with `pyannote/speaker-diarization-community-1` loaded from `DIARIZATION_PIPELINE_DIR` (ADR-001); exclusive turns, overlap kept separately, anonymous `SPEAKER_nn` ids, optional `--num-speakers` / `--min-speakers` / `--max-speakers`. The engine runs in a worker subprocess in its own process group with `--timeout`; Hugging Face libraries are forced offline and the worker gets no tokens. Audio is extracted with transcription-skill's fixed recipe (one explicit stream, mono 16 kHz PCM).
- `diarization assign`: transcription-skill transcript + turns -> the same transcript with `speaker_id` (per word when words exist, segment by majority duration; zero overlap or a tie stays null) and a `speaker_assignment` provenance block. The output passes `transcription check`.
- `diarization check` (turns validator), `diarization score` (frame-based DER), `diarization contract`, `diarization doctor` (MISSING with install steps; never downloads).
- Not yet measured: DER of the real engine on a two-voice recording (needs a machine with the gated weights).
