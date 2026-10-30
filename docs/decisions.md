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
