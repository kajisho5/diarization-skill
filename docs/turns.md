# Speaker turns (`diarization-skill/turns@1`)

JSON Schema: `schemas/turns.schema.json`. The validator in `src/diarization_skill/turns.py` is the executable form; `diarization check` runs it.

| field | meaning |
|---|---|
| `schema` | `diarization-skill/turns@1` |
| `source` | `filename` (bare name), `fingerprint` (`sha256:<hex>` of the file bytes, the same value as a transcription-skill transcript's `source.fingerprint`), `size_bytes`, `media_duration`, `has_video`, `audio_stream` (0-based stream decoded) |
| `engine` | `id`, `version` (pyannote.audio), `model` (`pyannote/speaker-diarization-community-1`), `revision` (snapshot commit of the pipeline directory, or null when it cannot be read), `license` (`CC-BY-4.0`), `attribution`, `execution_mode` (`local`), `device` |
| `parameters` | `num_speakers`, `min_speakers`, `max_speakers` as requested (null when not) |
| `speakers`, `speaker_count` | anonymous ids `SPEAKER_00`, `SPEAKER_01`, ... numbered by first appearance |
| `turns` | `[{start, end, speaker}]`, exclusive: ordered, non-overlapping, within the media duration. pyannote's `exclusive_speaker_diarization` |
| `speech_seconds` | total length of `turns` |
| `overlap` | `available`; `turns` (the engine's regular output, which may overlap); `regions` (`[{start, end, speakers}]` where two or more speak); `total_seconds` |
| `audio_extraction` | the fixed recipe (mono, 16000 Hz, pcm_s16le WAV), the `-map 0:a:N` stream, `engine_seconds` |
| `skill`, `skill_version`, `tool`, `created_at`, `processing_seconds`, `warnings` | provenance of the run |

Times are seconds on the media file's own timeline (no offset), rounded to milliseconds. An engine end up to 1 s past the media end is clamped with a warning; anything else wrong (overlapping "exclusive" turns, larger overruns, non-finite times) is rejected as `INVALID_RESULT`, never repaired. No path, command line or credential appears anywhere in the document.

## ASSIGN output

The input transcript, unchanged except for:

- `segments[].speaker_id`: an anonymous id or null.
- `segments[].words[].speaker_id` (when the segment has words): the same per word. An extra field; transcription-skill's validator accepts it.
- `warnings`: appended `speaker_id: ...` notes (segments left null, segments touching overlap, a source mismatch).
- `speaker_assignment`: `tool`, `skill_version`, `method`, `rule`, `engine` (with `revision`, `license`, `attribution`), `parameters`, `speakers`, `segments_per_speaker`, `counts`, `source_match`, `overlap_segment_ids`.

Rule: a word (or a segment without words) gets the speaker whose exclusive turns overlap it longest; zero overlap or a tie leaves null. A segment with words gets the speaker with the largest total duration of its assigned words; a tie or no assigned word leaves null. Nothing is inferred from neighbours.
