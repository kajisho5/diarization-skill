---
name: diarization-skill
description: Find who spoke when in a recording (anonymous speaker turns SPEAKER_00, SPEAKER_01, ...) and fill speaker_id in a transcription-skill transcript. Use when asked to split a talk, panel, interview or meeting by speaker, label transcript or subtitle lines by speaker, count speakers, find where people talk over each other, or suggest which programme name belongs to which speaker. Local and offline (pyannote.audio 4.x with the speaker-diarization-community-1 pipeline from a local directory); it never names anyone by itself: names are proposals from the caller's roster (consented voiceprints, lower-third OCR) that a person confirms.
---

# diarization-skill

Run `diarization <command> ... --json` (installed by `pip install -e .`; the engine by `pip install -e '.[pyannote]'`). Every command prints one JSON document. `diarization doctor --json` says whether DIARIZE is ready and, if not, what to install (it never downloads anything).

| Task | Command |
|---|---|
| Who spoke when | `diarization diarize talk.mp4 -o turns.json --json` |
| Same, speaker count known | add `--num-speakers 2` |
| Same, a range (panel, Q&A) | add `--min-speakers 3 --max-speakers 8` |
| A different audio track | add `--audio-stream 1` (0-based) |
| Put speakers into a transcript | `diarization assign transcript.json turns.json -o transcript.speakers.json --json` |
| Voiceprint of a consenting person | `diarization enroll alice.wav --name "Alice Smith" --consent-by "Alice Smith" --consent-date 2026-11-01 -o alice.voice.json --json` |
| Turns with voice vectors (for naming) | `diarization diarize panel.mp4 --embeddings -o turns.json --json` |
| Propose names from the programme | `diarization name turns.json --roster roster.json [--lower-thirds text.json] -o names.json --json` |
| Check a turns, voiceprint or proposals file | `diarization check file.json --json` |
| Measure against a hand-made reference | `diarization score turns.json reference.json --json` (DER) |
| What this skill declares | `diarization contract --json`, `diarization doctor --json` |

## When to use it

- After transcription-skill, when the transcript should say who is speaking (subtitles with speaker labels, minutes, dubbing per voice, cutting one speaker's answers).
- To count speakers or find speaker changes in a recording.
- Pass `--num-speakers` when you know it (an interview with two people). A wrong count is worse than none, so leave it out when unsure, or give a range.

## Report it honestly

- **Speaker ids are anonymous.** `SPEAKER_00` is "the first voice heard", not a person. Never write a name for a speaker id unless a human told you who it is, and then say the human said so. Ids from two different runs or files do not correspond.
- **It is an estimate.** Say how many speakers were found and that it is automatic. Short interjections, similar voices, music and far-away microphones cause errors. If the count is surprising, say so and suggest `--num-speakers`.
- **Overlap.** `turns` give one speaker at a time; when people talk over each other only one of them is in `turns`. `overlap.regions` / `overlap.total_seconds` say where and how much; mention it when it is more than a few seconds. ASSIGN lists the transcript segments that touch overlap in `speaker_assignment.overlap_segment_ids`.
- **Names are proposals.** `name` proposes a person from the caller's roster only, with its evidence (`basis`: `voice` similarity to a consented voiceprint, `lower_third` OCR text near the first turn). Show each proposal with its evidence and ask a person to confirm before you put a name in subtitles, chapters or anything published; record who confirmed. `anonymous` and `conflict` speakers keep their ids: do not pick a name for them. Never enroll a voice without the person's consent (`--consent-by`, `--consent-date`); a sample with other voices is refused. The default thresholds are not calibrated on the real engine yet: say so.
- **Null means unknown.** ASSIGN leaves `speaker_id` null when no speaker turn overlaps the words, or two speakers overlap them equally. Report how many segments stayed null (`counts.unassigned`); do not fill them in yourself.
- **Attribution.** The model is CC-BY-4.0. Every result carries `engine.attribution`; keep it with anything you publish from it.

Exit codes: `0` ok, `2` invalid input or arguments, `3` pyannote, the pipeline directory or ffmpeg missing (see `doctor`), `4` the engine failed or timed out.

## Not this skill's job

Transcription (transcription-skill), confirming a name (a human decision the agent records), identifying anyone outside the caller's roster, OCR of lower thirds (detection-skill `text`), separating overlapping voices, subtitle formatting (subtitle-skill), downloading model weights (a one-time manual step, see README).
