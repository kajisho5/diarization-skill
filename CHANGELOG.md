# Changelog

## 0.3.1 - 2026-11-13

The speaker-id rule as a spec with shared conformance fixtures (roadmap step 9, ADR-004). No behaviour change.

- `docs/speaker_id_rule.md`: the rule ASSIGN applies (turns -> `speaker_id`), owned by this repository.
- `conformance/speaker_id/` (`fixture_version` 1): six cases, each a transcript, a turns document and the expected `speaker_id` per segment and word plus `overlap_segment_ids`. They cover overlap, a word across a turn boundary, a gap with no turn, a word that touches a turn edge, ties and null words that do not vote. A manifest lists the SHA-256 of each case.
- `tests/test_speaker_id_conformance.py`: ASSIGN passes every case; the fixture hashes are pinned per `fixture_version`; with a transcription-skill checkout, `transcription speakers` passes every case too.

## 0.3.0 - 2026-11-06

Long recordings (roadmap step 8, ADR-003).

- `diarization diarize --chunk-seconds S`: diarizes in equal chunks, each extracted on its own (the fixed recipe with `-ss/-t`), with `--timeout` per chunk. Speakers are linked across chunks by cosine similarity of their embeddings to each speaker's running centroid, one to one per chunk, with `--link-threshold` (default 0.5). The result has `chunking` (chunks, and links with similarity and margin).
- `--work-dir`: each finished chunk is saved with a key and reused on a rerun (resumable). A chunk file from another run is refused, never overwritten.

## 0.2.0 - 2026-11-06

Names for speakers (roadmap step 7, ADR-002).

- `diarization enroll`: a voiceprint (`diarization-skill/voiceprint@1`) of one consenting person from a sample of their voice alone. Refused without `--consent-by` / `--consent-date`, and when the sample holds more than one voice (`ENROLLMENT_REJECTED`).
- `diarization diarize --embeddings`: the turns carry one speaker embedding per anonymous id (`speaker_embeddings`; pyannote's own centroids). Without the flag no voice vector is written.
- `diarization name`: name proposals (`diarization-skill/name-proposals@1`) from the caller's roster (`diarization-skill/roster@1`): voice similarity with a margin over other people and speakers, and detection-skill lower-third OCR text near a speaker's first turn. Conflicts and missing evidence leave the id anonymous; every proposal is unconfirmed.
- `diarization check` validates turns, voiceprints and name proposals by their `schema`. New error codes: `INVALID_ROSTER`, `INVALID_VOICEPRINT`, `INVALID_DETECTIONS`, `EMBEDDING_MISMATCH`, `ENROLLMENT_REJECTED`.

## 0.1.0 - 2026-10-30

First release (roadmap steps 2-4).

- `diarization diarize`: media -> speaker turns (`diarization-skill/turns@1`, `schemas/turns.schema.json`). pyannote.audio 4.x with `pyannote/speaker-diarization-community-1` loaded from `DIARIZATION_PIPELINE_DIR` (ADR-001); exclusive turns, overlap kept separately, anonymous `SPEAKER_nn` ids, optional `--num-speakers` / `--min-speakers` / `--max-speakers`. The engine runs in a worker subprocess in its own process group with `--timeout`; Hugging Face libraries are forced offline and the worker gets no tokens. Audio is extracted with transcription-skill's fixed recipe (one explicit stream, mono 16 kHz PCM).
- `diarization assign`: transcription-skill transcript + turns -> the same transcript with `speaker_id` (per word when words exist, segment by majority duration; zero overlap or a tie stays null) and a `speaker_assignment` provenance block. The output passes `transcription check`.
- `diarization check` (turns validator), `diarization score` (frame-based DER), `diarization contract`, `diarization doctor` (MISSING with install steps; never downloads).
- Not yet measured: DER of the real engine on a two-voice recording (needs a machine with the gated weights).
