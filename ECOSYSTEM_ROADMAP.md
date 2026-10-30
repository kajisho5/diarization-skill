# Roadmap — diarization-skill

Part of the [AI Video Production OS](https://github.com/kajisho5/AI-video-production-OS/blob/main/docs/ecosystem/ROADMAP.md) ecosystem: department **認識 (Recognition)**, status `scaffold`.
Written from a read-only audit of this repository on 2026-10-30. Each step below is meant to be one pull request; tick it off by linking the PR.

## Where it stands

Scaffold. Engine chosen (ADR-001: pyannote.audio 4.x, community-1 pipeline, offline after one gated download); no code yet.

## What this repository owns

`ecosystem.json` gives each job to exactly one repository. Work outside this list belongs to its owner and is handed over, not re-implemented.

| Owned job | Status | Note |
|---|---|---|
| 話者識別 | missing |  |

## Wiring

None. The agent hard-codes speaker_id = null in about 5 places; transcription reserves the field.

## Steps

- [x] **1. Engine ADR** ([docs/decisions.md](docs/decisions.md) ADR-001: pyannote.audio 4.x + speaker-diarization-community-1) — candidates to verify: pyannote.audio (gated weights), NeMo Sortformer/MSDD, WeSpeaker, 3D-Speaker, SpeechBrain -- licence, offline after one pull, CPU speed
  - Done when: ADR merged
- [ ] **2. Skeleton + speaker-turns schema + contract/doctor**
  - Done when: OS static checks PASS
- [ ] **3. DIARIZE op** — media -> anonymous turns, optional num_speakers
  - Done when: two-voice fixture, DER under threshold
- [ ] **4. ASSIGN op** — transcript + turns -> transcript with speaker_id (transcription schema)
  - Done when: round trip through `transcription check`
- [ ] **5. SKILL.md, CI; (agent side) SkillSpec + adapter**
  - Done when: tracked in video-production-agent

---
Engine for every media operation: [ffmpeg-skill](https://github.com/kajisho5/ffmpeg-skill) (42 tools, treated as complete). Orchestrator: [video-production-agent](https://github.com/kajisho5/video-production-agent).
