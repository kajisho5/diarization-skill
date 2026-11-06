# Speaker names: voiceprints, roster and name proposals (ADR-002)

JSON Schemas: `schemas/voiceprint.schema.json`, `schemas/roster.schema.json`, `schemas/name-proposals.schema.json`; the validators in `src/diarization_skill/voices.py` and `naming.py` are the executable form, and `diarization check` picks the right one by `schema`.

```bash
diarization enroll alice.wav --name "Alice Smith" --consent-by "Alice Smith" --consent-date 2026-11-01 \
    --consent-note "speaker naming, Example Conference 2026" -o alice.voice.json
diarization diarize panel.mp4 --embeddings -o turns.json
detection text panel.mp4 --every 1 -o text.json              # optional: lower thirds (detection-skill)
diarization name turns.json --roster roster.json --lower-thirds text.json -o names.json --json
```

## Voiceprint (`diarization-skill/voiceprint@1`)

| field | meaning |
|---|---|
| `person` | `name` (as in the roster), `person_id`, `role` |
| `consent` | `given_by`, `date` (YYYY-MM-DD), `note`; required |
| `sample` | `filename` (bare), `fingerprint`, `size_bytes`, `media_duration`, `audio_stream`, `speakers_found`, `speech_seconds`, `dominant_seconds`, `dominant_share` |
| `engine` | `id`, `version`, `model`, `revision`, `license`, `attribution` of the engine that made the embedding |
| `embedding` | `metric: cosine`, `dimension`, `vector` |
| `parameters` | `min_share`, `min_speech` used to accept the sample |

## Roster (`diarization-skill/roster@1`)

```json
{"schema": "diarization-skill/roster@1", "event": "Example Conference 2026, panel 2",
 "people": [{"id": "chair", "name": "Alice Smith", "role": "chair", "voiceprint": "alice.voice.json"},
            {"id": "p1", "name": "Taro Yamada", "role": "speaker", "aliases": ["山田 太郎"]}]}
```

`voiceprint` paths are relative to the roster file; its `person.name` must equal the roster `name`.

## Name proposals (`diarization-skill/name-proposals@1`)

| field | meaning |
|---|---|
| `requires_confirmation` | always `true` |
| `turns`, `roster`, `lower_thirds` | the inputs: bare file names, sha256 of the files, the turns' source fingerprint and engine; voiceprint files with their consent |
| `method` | `voice` (`metric`, `min_similarity`, `min_margin`), `lower_third` (`window_before`, `window_after`, `match`), `one_person_per_speaker` |
| `speakers` | one row per anonymous speaker: `speaker`, `first_turn_start`, `status` (`proposed` / `anonymous` / `conflict`), `proposal` (`person_id`, `name`, `role`, `basis` [`voice`, `lower_third`], `similarity`, `confirmed: false`) or null, `evidence` (`voice`: every person's similarity, best first; `lower_third`: every matching text with its time and `seconds_from_first_turn`), `reasons` |
| `unmatched_people` | roster ids proposed for nobody |
| `counts` | `proposed`, `anonymous`, `conflict`, `speakers` |

Rules (checked by the validator): ids stay `SPEAKER_nn`; a proposal exists exactly when `status` is `proposed`, names its basis and is unconfirmed; one person at most once; a speaker without a proposal has a reason.
