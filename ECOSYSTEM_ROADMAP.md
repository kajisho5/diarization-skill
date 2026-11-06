# Roadmap — diarization-skill

Part of the [AI Video Production OS](https://github.com/kajisho5/AI-video-production-OS/blob/main/docs/ecosystem/ROADMAP.md) ecosystem: department **認識 (Recognition)**, status `scaffold`.
Written from a read-only audit of this repository on 2026-10-30. Each step below is meant to be one pull request; tick it off by linking the PR.

## Where it stands

Early (0.3.0). DIARIZE (anonymous speaker turns, `diarization-skill/turns@1`) and ASSIGN (speaker_id into a transcription-skill transcript) work, with contract and doctor. NAME proposes programme names for the anonymous speakers (consented voiceprints, lower thirds) for a person to confirm. Long recordings run in chunks with speakers linked across them, resumable. The real engine's DER is not measured yet: that needs a machine with the gated weights.

## What this repository owns

`ecosystem.json` gives each job to exactly one repository. Work outside this list belongs to its owner and is handed over, not re-implemented.

| Owned job | Status | Note |
|---|---|---|
| 話者識別 | partial | `diarize` + `assign` (0.1.0); name proposals from a roster: `enroll` + `name` (0.2.0); real-engine DER and naming thresholds pending |

## Wiring

transcription-skill transcript -> `diarization assign` -> the same transcript with `speaker_id` (passes `transcription check`). The agent measures DIARIZE as `speaker_turns` (step 5, video-production-agent#98) and, with `speakers.assign=true`, carries the speaker ids into subtitles and take picks through transcription-skill's merge (video-production-agent#111, its ADR-096).

## Steps

- [x] **1. Engine ADR** ([docs/decisions.md](docs/decisions.md) ADR-001: pyannote.audio 4.x + speaker-diarization-community-1) — candidates to verify: pyannote.audio (gated weights), NeMo Sortformer/MSDD, WeSpeaker, 3D-Speaker, SpeechBrain -- licence, offline after one pull, CPU speed
  - Done when: ADR merged
- [x] **2. Skeleton + speaker-turns schema + contract/doctor** (0.1.0: `schemas/turns.schema.json`, `diarization contract` / `doctor`; publishes_contract, lifecycle_declared, dependency_version_ranges, no_unsafe_shell_out and doctor_status PASS in `tests/`)
  - Done when: OS static checks PASS
- [x] **3. DIARIZE op** — media -> anonymous turns, optional num_speakers (0.1.0: op done; the synthetic two-voice fixture passes end to end with a test engine, DER 0. **Pending:** the real engine's DER on a two-voice recording, which needs a machine with the weights: set `DIARIZATION_PIPELINE_DIR`, `DIARIZATION_DER_MEDIA`, `DIARIZATION_DER_REFERENCE` and run the tests, or `diarization score`)
  - Done when: two-voice fixture, DER under threshold
- [x] **4. ASSIGN op** — transcript + turns -> transcript with speaker_id (transcription schema) (0.1.0: `tests/` runs `transcription check` on the output)
  - Done when: round trip through `transcription check`
- [x] **5. SKILL.md, CI; (agent side) SkillSpec + adapter** (skill side done in 0.1.0, checked in [#5](https://github.com/kajisho5/diarization-skill/pull/5): `SKILL.md` documents every CLI command (`diarize`, `assign`, `check`, `score`, `contract`, `doctor`) with examples; `.github/workflows/tests.yml` (unit: Linux / macOS / Windows x Python 3.9 / 3.12 plus `diarization contract --json`; integration: ffmpeg + transcription-skill + the OS checkout, only the real-engine tests may skip, `doctor`); the OS checkout now names `kajisho5/Velisa`. Agent side: [video-production-agent#98](https://github.com/kajisho5/video-production-agent/pull/98) (its ADR-086) wires DIARIZE as the analysis kind `speaker_turns`, with a handoff when DIARIZE cannot run)
  - Done when: tracked in video-production-agent
- [ ] **6. Measured on Japanese conference recordings** — DER on consented recordings of a panel, a Q&A with floor microphones and a two-person interview, with and without overlap, committed as a report. Field review 2026-11-05 (Velisa ROADMAP, third round)
  - なぜ要るか：質疑応答や座談会で使えるかを、数字で判断できる
  - Done when: a report with DER per condition from a machine with the weights
- [x] **7. Names for speakers** — anonymous speakers mapped to names from the caller's list (the programme: chair and speakers) using enrolled voice samples given with consent, or a lower third's OCR text (detection-skill) near the speaker's first turn; every mapping is a proposal with its evidence, never silent (Done (ADR-002): 0.2.0 `diarization enroll` (voiceprint with a required consent record; a sample with other voices is refused), `diarize --embeddings`, `diarization name` (`diarization-skill/name-proposals@1`: voice similarity with a margin and detection-skill lower thirds near the first turn, `conflict` / `anonymous` instead of a guess, every proposal `confirmed: false`). `tests/test_naming.py` runs the fixture through the real extraction with a test engine that has embeddings. **Pending:** calibrating `--min-similarity` / `--min-margin` on pyannote's embeddings, which needs consented recordings and the weights (with step 6))
  - なぜ要るか：字幕の話者名や章立てに、実名を手で当てる手間を減らせる
  - Done when: a fixture with two enrolled voices maps both, and an unenrolled voice stays anonymous
- [x] **8. Long recordings** — 2-8 h diarized in chunks with speakers linked across chunks by their embeddings, resumable, with the link confidence in the result (Done (ADR-003): 0.3.0 `diarize --chunk-seconds` (equal chunks, each extracted on its own, timeout per chunk; one-to-one linking by cosine similarity to running centroids; `chunking.links` with similarity and margin, `min_link_similarity`) and `--work-dir` (finished chunks saved with a key and reused on a rerun). `tests/test_chunks.py` runs the fixture through the real per-chunk extraction with a test engine that has embeddings. **Pending:** a 2-8 h recording through pyannote and calibrating `--link-threshold`, which need the weights and real recordings (with step 6))
  - なぜ要るか：終日のセッションでも、同じ座長が途中で別人扱いにならない
  - Done when: a fixture split in three chunks keeps the same speaker id for the same voice across chunks

---
Engine for every media operation: [ffmpeg-skill](https://github.com/kajisho5/ffmpeg-skill) (3.x, 42 tools, contract_version 2.0; treated as complete, proposals go to Velisa's ROADMAP). Orchestrator: [video-production-agent](https://github.com/kajisho5/video-production-agent).
