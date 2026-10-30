# Roadmap — diarization-skill

Part of the [AI Video Production OS](https://github.com/kajisho5/AI-video-production-OS/blob/main/docs/ecosystem/ROADMAP.md) ecosystem: department **認識 (Recognition)**, status `scaffold`.
Written from a read-only audit of this repository on 2026-10-30. Each step below is meant to be one pull request; tick it off by linking the PR.

## Where it stands

Early (0.1.0). DIARIZE (anonymous speaker turns, `diarization-skill/turns@1`) and ASSIGN (speaker_id into a transcription-skill transcript) work, with contract and doctor. The real engine's DER is not measured yet: that needs a machine with the gated weights.

## What this repository owns

`ecosystem.json` gives each job to exactly one repository. Work outside this list belongs to its owner and is handed over, not re-implemented.

| Owned job | Status | Note |
|---|---|---|
| 話者識別 | partial | `diarize` + `assign` (0.1.0); real-engine DER pending |

## Wiring

transcription-skill transcript -> `diarization assign` -> the same transcript with `speaker_id` (passes `transcription check`). The agent still hard-codes speaker_id = null in about 5 places (step 5).

## Steps

- [x] **1. Engine ADR** ([docs/decisions.md](docs/decisions.md) ADR-001: pyannote.audio 4.x + speaker-diarization-community-1) — candidates to verify: pyannote.audio (gated weights), NeMo Sortformer/MSDD, WeSpeaker, 3D-Speaker, SpeechBrain -- licence, offline after one pull, CPU speed
  - Done when: ADR merged
- [x] **2. Skeleton + speaker-turns schema + contract/doctor** (0.1.0: `schemas/turns.schema.json`, `diarization contract` / `doctor`; publishes_contract, lifecycle_declared, dependency_version_ranges, no_unsafe_shell_out and doctor_status PASS in `tests/`)
  - Done when: OS static checks PASS
- [x] **3. DIARIZE op** — media -> anonymous turns, optional num_speakers (0.1.0: op done; the synthetic two-voice fixture passes end to end with a test engine, DER 0. **Pending:** the real engine's DER on a two-voice recording, which needs a machine with the weights: set `DIARIZATION_PIPELINE_DIR`, `DIARIZATION_DER_MEDIA`, `DIARIZATION_DER_REFERENCE` and run the tests, or `diarization score`)
  - Done when: two-voice fixture, DER under threshold
- [x] **4. ASSIGN op** — transcript + turns -> transcript with speaker_id (transcription schema) (0.1.0: `tests/` runs `transcription check` on the output)
  - Done when: round trip through `transcription check`
- [ ] **5. SKILL.md, CI; (agent side) SkillSpec + adapter**
  - Done when: tracked in video-production-agent

---
Engine for every media operation: [ffmpeg-skill](https://github.com/kajisho5/ffmpeg-skill) (42 tools, treated as complete). Orchestrator: [video-production-agent](https://github.com/kajisho5/video-production-agent).
