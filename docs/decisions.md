# Architecture Decision Records

## ADR-001 Engine: pyannote.audio 4.x with the `speaker-diarization-community-1` pipeline

Checked 2026-10-30 against the model cards and project pages listed under Sources.

### Requirements

The skill runs in the field (conference and event recordings, hybrid sessions), so the engine has to meet these:

- **Offline.** It runs offline after one model download. No cloud calls at run time.
- **Licence.** The licence allows commercial use of the output. Attribution is acceptable.
- **CPU.** It runs on a laptop CPU. A GPU is optional.
- **Speakers and language.** It handles more than four speakers (panels, Q&A) and is not limited to English.
- **Transcript fit.** Its output can be joined to a transcription-skill transcript (segments and words with times).

### Candidates

| Candidate | Licence (weights) | Offline | Limits found | Verdict |
|---|---|---|---|---|
| **pyannote.audio 4.x + `pyannote/speaker-diarization-community-1`** | CC-BY-4.0 ("will always remain freely accessible") | Yes. Gated: accept the user conditions once, clone, then load from a local path "without internet connection" | CPU by default. The card gives no CPU real-time factor, so we measure it ourselves (step 3) | **Chosen** |
| sherpa-onnx (ONNX): pyannote segmentation-3.0 + a speaker-embedding model | segmentation-3.0: MIT. Embedding models: licence per model, not stated on the sherpa-onnx page | Yes, and no PyTorch | The embedding licence must be checked per model | **Second backend, later.** For machines where PyTorch is too heavy |
| NVIDIA Streaming Sortformer 4spk v2.1 | NVIDIA Open Model License Agreement | Yes (NeMo) | Up to 4 speakers ("degrades with 5 or more"). Trained mainly on English. Speed figures are for GPU only. Needs NeMo | Rejected: speaker cap, English bias, heavy dependency |
| Sortformer 4spk v1 | CC-BY-NC-4.0 | Yes | Non-commercial | Rejected: licence |
| Rev `reverb-diarization-v1` | Non-commercial | Yes | Non-commercial | Rejected: licence |
| WeSpeaker / 3D-Speaker as full pipelines | Code is Apache-2.0. Weights follow the training data licence (VoxCeleb models: CC-BY-4.0) | Yes | Would mean assembling segmentation, embedding and clustering ourselves | Not now. Their embeddings can still feed the sherpa-onnx backend |

### Decision

- **Default engine.** The engine is `pyannote.audio` 4.x, with the community-1 pipeline loaded from a local directory. The skill never downloads anything at run time.
- **Doctor.** `doctor` reports `MISSING` until the pipeline directory exists, and prints how to get it:
  1. Accept the user conditions on Hugging Face.
  2. `git lfs` clone the pipeline once.
  3. Point `DIARIZATION_PIPELINE_DIR` at the clone.
- **Output.** The skill uses `exclusive_speaker_diarization`, which allows one speaker at a time. ASSIGN uses that, since a transcript segment gets one `speaker_id`. Overlapping speech is still available from the regular output, and is kept in provenance.
- **Speaker ids.** They are anonymous (`SPEAKER_00`, ...). Naming a speaker is a human decision that the agent records. It is never inferred.
- **Attribution.** CC-BY-4.0 attribution for the pipeline goes in the README, and in the provenance of every result: model id plus snapshot revision.
- **Engine process.** The engine runs in a worker subprocess with a timeout. This is the same pattern as transcription-skill ADR-003, because a long PyTorch call cannot be interrupted in-process.
- **Audio input.** One audio stream is extracted to mono 16 kHz PCM, with the same fixed recipe as transcription-skill ADR-004, so the two skills see the same samples.
- **Tests.** CI does not download gated weights. Unit tests use a fake engine. The real-engine tests are skipped unless the pipeline directory is present. DER on the two-voice fixture is measured locally and recorded in the step 3 PR.

### Follow-ups

- **Second backend.** Add sherpa-onnx as a second registered engine once its embedding model's licence is verified.
- **Prefer an engine.** Add `--engine` to choose between the two.

### Sources

- [pyannote/speaker-diarization-community-1 (model card)](https://huggingface.co/pyannote/speaker-diarization-community-1)
- [pyannote community-1 announcement](https://pyannote.ai/blog/community-1)
- [nvidia/diar_streaming_sortformer_4spk-v2.1](https://huggingface.co/nvidia/diar_streaming_sortformer_4spk-v2.1)
- [nvidia/diar_sortformer_4spk-v1](https://huggingface.co/nvidia/diar_sortformer_4spk-v1)
- [sherpa-onnx speaker diarization models](https://k2-fsa.github.io/sherpa/onnx/speaker-diarization/models.html)
- [pyannote/wespeaker-voxceleb-resnet34-LM (weights licence follows the training data)](https://huggingface.co/pyannote/wespeaker-voxceleb-resnet34-LM)

## ADR-002 Names for speakers: proposals from consented voiceprints and lower thirds, confirmed by a person

Roadmap step 7. Decided 2026-11-06.

### Context

Subtitles and chapters want real names, but DIARIZE only knows anonymous voices (ADR-001: naming is a human decision). Two kinds of evidence are usually at hand for a conference recording: the programme (who chairs, who speaks), and the lower thirds the video shows when someone starts talking. Some speakers may also agree to give a short sample of their voice. The point is to cut the manual work of matching ids to names, without ever putting a name on a voice by guessing.

### Decision

- **Three documents, one op each.**
  - `diarization enroll <sample>` writes a voiceprint (`diarization-skill/voiceprint@1`): the speaker embedding of the sample's one voice, made by the same engine, extraction and worker as DIARIZE.
  - `diarization diarize --embeddings` adds `speaker_embeddings` to the turns: one vector per anonymous id. For pyannote this is the pipeline's own `speaker_embeddings` (pyannote.audio 4.x, rows in `speaker_diarization.labels()` order; zero rows left out).
  - `diarization name <turns> --roster <roster>` writes name proposals (`diarization-skill/name-proposals@1`). It is pure computation, needs no engine, and never changes the turns or a transcript.
- **Consent first.** `enroll` refuses to run without `--consent-by` and `--consent-date`, and the record goes into the voiceprint. `name` refuses a voiceprint without it. A sample holding more than one voice is refused (`ENROLLMENT_REJECTED`): the main voice needs 80% of the speech and 3 s. Nothing is averaged or guessed.
- **Only the caller's roster.** A speaker can only be proposed as someone on the roster the caller gave. Nobody outside the list is identified, and there is no search against other voices.
- **Voice evidence.** This is cosine similarity between a speaker's vector and a voiceprint. Speaker S is proposed as person P only when all three hold:
  - the similarity is at least `--min-similarity` (default 0.6);
  - P beats every other person for S by `--min-margin` (default 0.1);
  - S beats every other speaker for P by the same margin.

  Embeddings compare only within one engine model. A different model, a different known snapshot or a different dimension is refused (`EMBEDDING_MISMATCH`). An unknown snapshot gives a warning.
- **Lower-third evidence.** This is detection-skill's OCR text (`detection text` detections, or its text `tracks`) read from the same file (the fingerprint must match, as for ASSIGN). It counts when the text falls between `--window-before` (default 5 s) before the speaker's first turn and `--window-after` (default 15 s) after it. It must contain a roster name or alias, compared after NFKC normalisation, case folding and dropping spaces and punctuation. So `山田　太郎` matches `山田太郎`, and `ＡＬＩＣＥ` matches `Alice`. There is no locale-specific rule. Other scripts or name orders go in `aliases`. Two names in one window give no proposal.
- **Never silent.** Every speaker row carries the full evidence (all similarities, every matching lower third) and reasons. A person is proposed for at most one speaker. When voice and lower third disagree, or two speakers point at the same person, the speaker is `conflict` and gets no proposal. With no evidence it stays `anonymous`. Every proposal has `confirmed: false` and the document `requires_confirmation: true`. Confirming is a person's decision, which the agent records. This skill has no command that confirms.
- **Voice vectors are personal data.** Turns carry no embeddings unless `--embeddings` is given. Voiceprints carry the person's name and consent and should be kept with the consent form. No path appears in any of the three documents.

### Consequences

- The defaults are not calibrated on the real engine yet: they were set on the synthetic fixture. Calibrating them needs consented recordings and a machine with the weights (with step 6).
- The roadmap fixture runs through the real extraction with a test engine that has embeddings (`tests/test_naming.py`). Two enrolled voices are proposed by name, and an unenrolled third voice stays anonymous.

## ADR-003 Long recordings: chunks linked by speaker embeddings, resumable

Roadmap step 8. Decided 2026-11-06.

### Context

A full conference day is 2-8 h. One engine pass over that length is slow and holds the whole waveform in memory. A crash at hour 7 loses everything. The anonymous ids must not change halfway either: the chair at 09:00 has to be the same id as the chair at 16:00.

### Decision

- **Opt-in.** `diarize --chunk-seconds S` turns it on (for example 1800); without it nothing changes. The recording is split into `ceil(duration / S)` equal chunks.
- **Extraction.** Each chunk is extracted on its own with the fixed recipe plus `-ss start -t length` before `-i`, so there is no single multi-hundred-MB WAV.
- **Engine.** The engine runs on each chunk in the worker, and `--timeout` applies per chunk.
- **Speaker counts.** For each chunk, `--num-speakers N` becomes `max_speakers N`, since a chunk may hold fewer people. `--min-speakers` is not passed down. `--max-speakers` is.
- **Linking.** It uses the engine's speaker embeddings (pyannote 4.x `speaker_embeddings`, ADR-002). An engine without embeddings is refused for chunked runs.
  - Chunk 0's speakers become the first global speakers.
  - For each later chunk, every chunk speaker is compared with every global speaker's centroid: the speech-weighted sum of its chunk vectors, by cosine similarity.
  - Pairs at or above `--link-threshold` (default 0.5) are taken greedily, highest first, one to one. Two speakers in one chunk are two people by the engine's own decision.
  - A chunk speaker that is not linked, or has no vector, becomes a new id.
- **Turns.** They are shifted to the media timeline, clipped to their chunk, and a turn cut by a boundary is joined again.
- **Link confidence in the result.** `chunking.links` has one row per chunk speaker: `chunk`, `chunk_speaker`, `speaker` (the document's id), `linked`, `similarity` (to the chosen speaker, or the best one when not linked) and `margin` (over the next-best speaker). `chunking.min_link_similarity` is the weakest link. When linking leaves more ids than `--num-speakers`, a warning says so instead of merging anyone by force.
- **Resumable.** With `--work-dir D`, each finished chunk's engine result is saved as `D/chunk_NNNN.json`, with a key over the source fingerprint, stream, engine, model, snapshot, speaker counts and chunk bounds.
  - A rerun reuses every chunk with a matching key, and `chunks[].resumed` says which.
  - A file with another key is refused, never overwritten.
  - Linking is recomputed from the saved results, so `--link-threshold` can change between runs.
  - The work dir holds voice vectors: it is the caller's to keep or delete.

### Consequences

- The roadmap fixture (`tests/test_chunks.py`) runs through the real per-chunk extraction with a test engine that has embeddings. It is split into three chunks, and each voice keeps one id across them, including a turn that crosses a boundary. The same fixture also shows the run resuming after a failure in chunk 1.
- The default `--link-threshold` is not calibrated on pyannote's embeddings, and no 2-8 h recording has been run with the real engine. Both need the weights and real recordings (with step 6).
- A boundary that falls inside a word can split it between two chunks. The joined turn keeps the times, but the engine saw two halves. Overlapping chunks are a possible follow-up if this shows up in measurements.
