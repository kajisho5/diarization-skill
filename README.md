<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="brand/app-icon-dark.png">
    <img src="brand/app-icon.png" width="128" alt="diarization-skill app icon">
  </picture>
</p>

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="brand/logo/diarization-skill-wordmark-white.svg">
    <img src="brand/logo/diarization-skill-wordmark-black.svg" height="48" alt="diarization-skill">
  </picture>
</p>

<p align="center"><b>Who spoke when, in the transcript</b></p>


Who spoke when, for the [AI Video Production OS](https://github.com/kajisho5/AI-video-production-OS) (department 認識 / Recognition). `diarize` turns a recording into anonymous speaker turns (`SPEAKER_00`, `SPEAKER_01`, ...). `assign` writes those speakers into a [transcription-skill](https://github.com/kajisho5/transcription-skill) transcript's `speaker_id`, the field transcription-skill reserves for this.

Local and offline: the engine is [pyannote.audio](https://github.com/pyannote/pyannote-audio) 4.x with the [`pyannote/speaker-diarization-community-1`](https://huggingface.co/pyannote/speaker-diarization-community-1) pipeline, loaded from a directory on disk ([ADR-001](docs/decisions.md)). Nothing is downloaded at run time.

```bash
pip install -e '.[pyannote]'          # the skill + pyannote.audio 4.x (Python >= 3.10 for the engine)
diarization doctor                    # what is missing, and how to get it
diarization diarize talk.mp4 --num-speakers 2 -o turns.json
diarization assign transcript.json turns.json -o transcript.speakers.json
```

`assign` and `check` need only the Python standard library (`pip install -e .`, Python >= 3.9).

## One-time setup: the pipeline weights

The weights are free (CC-BY-4.0) but gated: Hugging Face asks you to accept the user conditions once. After that the skill reads a local copy and never goes online.

1. Sign in to Hugging Face and accept the conditions at <https://huggingface.co/pyannote/speaker-diarization-community-1>.
2. Clone the pipeline once with Git LFS (Git asks for your user name and an access token with read scope; the skill never sees the token):
   ```bash
   git lfs install
   git clone https://huggingface.co/pyannote/speaker-diarization-community-1 ~/models/speaker-diarization-community-1
   ```
3. Point the skill at it:
   ```bash
   export DIARIZATION_PIPELINE_DIR=~/models/speaker-diarization-community-1
   diarization doctor    # pipeline: OK, with the snapshot revision
   ```

`doctor` reports `MISSING` until this is done, and `INCOMPLETE` when the weight files are still Git LFS pointers (run `git lfs pull` in the directory). The revision (the clone's commit) goes into every result. The worker that runs pyannote gets the Hugging Face libraries switched offline and no tokens in its environment.

## Commands

| Command | What it does |
|---|---|
| `diarize <media> [--num-speakers N \| --min-speakers N --max-speakers N] [--audio-stream 0] [--embeddings] [--chunk-seconds 1800 [--link-threshold 0.5] [--work-dir DIR]] [--pipeline-dir DIR] [--device cpu] [--timeout 3600] [-o turns.json]` | Speaker turns ([`diarization-skill/turns@1`](docs/turns.md)) |
| `assign <transcript.json> <turns.json> -o out.json [--replace] [--allow-source-mismatch]` | Transcript with `speaker_id` filled and a `speaker_assignment` provenance block |
| `enroll <sample> --name NAME --consent-by WHO --consent-date YYYY-MM-DD [--consent-note TEXT] -o voice.json` | Voiceprint of one consenting person ([ADR-002](docs/decisions.md), [docs/naming.md](docs/naming.md)) |
| `name <turns.json> --roster roster.json [--lower-thirds text.json] [--min-similarity 0.6] [--min-margin 0.1] -o names.json` | Name proposals for the anonymous speakers, each with its evidence; never confirmed by the tool |
| `check <file.json>` | Validate a turns, voiceprint or name-proposals document |
| `score <turns.json> <reference.json> [--with-overlap]` | Diarization error rate against a reference (frame-based, no collar) |
| `contract --json` | Machine-readable contract in the OS registry's shape |
| `doctor --json` | Python, ffmpeg, pyannote.audio, torch, the pipeline directory; `MISSING` entries carry install steps |

Every command prints one JSON document with `--json`. Exit codes: 0 ok, 2 invalid input or arguments, 3 engine, pipeline or ffmpeg missing, 4 the engine failed or timed out. Outputs never overwrite an existing file without `--overwrite`, and `-o` equal to an input is refused.

## How it works

- **Audio.** ffprobe checks the input, then ffmpeg extracts one explicitly mapped audio stream to mono 16 kHz PCM, the same fixed recipe as transcription-skill (its ADR-004), so both skills see the same samples on the same timeline. The input file is fingerprinted (sha256); ASSIGN refuses turns whose fingerprint differs from the transcript's unless you say both share a timeline (`--allow-source-mismatch`).
- **Engine.** pyannote runs in a worker subprocess (`python -m diarization_skill.engines.worker`) in its own process group, killed on `--timeout`. The audio is handed over as a waveform, so pyannote does not decode media itself.
- **Turns.** `turns` are pyannote's `exclusive_speaker_diarization` (one speaker at a time). The regular output, where people may overlap, is kept under `overlap` with the regions where two or more speak. Ids are renumbered `SPEAKER_00`, `SPEAKER_01`, ... by first appearance. They name nobody: naming a speaker is a human decision.
- **Assign.** For every word (or a segment without words), the speaker whose exclusive turns overlap it longest. No overlap, or a tie, stays `null`, and nothing is guessed from neighbours. A segment with words takes the speaker with the largest total duration of its words. Everything else in the transcript is unchanged, and the result passes `transcription check`. The rule is written down in [docs/speaker_id_rule.md](docs/speaker_id_rule.md), with conformance fixtures in [`conformance/speaker_id/`](conformance/speaker_id/) that transcription-skill's speaker merge runs too ([ADR-004](docs/decisions.md)).

- **Long recordings.** `--chunk-seconds 1800` diarizes a 2-8 h recording in equal chunks (each extracted on its own, `--timeout` per chunk) and links each chunk's speakers to the earlier ones by cosine similarity of their embeddings, so one voice keeps one id all day. Every link, with its similarity and margin, is in `chunking.links`. `--work-dir` saves each finished chunk and a rerun picks up where it stopped ([ADR-003](docs/decisions.md)).
- **Names.** `name` proposes, for each anonymous speaker, a person from the roster you give (the programme). Evidence is the cosine similarity of the speaker's embedding (`diarize --embeddings`) to a consented voiceprint (`enroll`), and detection-skill OCR text naming the person near the speaker's first turn. A proposal needs a clear margin over every other person and speaker; disagreements are `conflict`, no evidence is `anonymous`, and every proposal is `confirmed: false` until a person confirms it ([ADR-002](docs/decisions.md)).

## Limits

- The DER of the real engine on the two-voice fixture is not measured yet; it needs a machine with the weights (`ECOSYSTEM_ROADMAP.md`, step 3). `diarization score` and the `DIARIZATION_DER_*` test do the measurement.
- CPU speed is not measured yet either. The model card gives no CPU real-time factor.
- Overlapping speech gets one speaker in `turns`. Similar voices, short interjections, music and distant microphones cause errors.
- The naming thresholds (`--min-similarity 0.6`, `--min-margin 0.1`) were set on a synthetic fixture and are not calibrated on pyannote's embeddings yet.
- The chunk-linking threshold (`--link-threshold 0.5`) is not calibrated on pyannote's embeddings, and no 2-8 h recording has been run with the real engine yet.
- One engine for now; a sherpa-onnx backend (no PyTorch) is a follow-up of ADR-001.

## Tests

```bash
python -m unittest discover -s tests -v
```

Unit tests inject test-only engines (`tests/fake_engine.py`, never registered in the package). They cover the schema, ASSIGN, the CLI, doctor, the worker subprocess and its timeout kill, and an end-to-end run through the real extraction on a synthetic two-voice fixture. `tests/test_speaker_id_conformance.py` runs ASSIGN on every speaker-id conformance case and checks the fixture hashes against the pinned `fixture_version`. The round trip through `transcription check`, and `transcription speakers` on the conformance cases, run when a transcription-skill checkout is found (`TRANSCRIPTION_SKILL_DIR` or `../transcription-skill`), and the OS registry conformance checks run when an AI-video-production-OS checkout is (`AI_VIDEO_PRODUCTION_OS_DIR` or `../AI-video-production-OS`). Real-engine tests run only when `DIARIZATION_PIPELINE_DIR` and pyannote.audio are present. To measure DER on your own recording, set `DIARIZATION_DER_MEDIA` and `DIARIZATION_DER_REFERENCE` (a JSON file with `turns: [{start, end, speaker}]`; `DIARIZATION_DER_THRESHOLD`, default 0.25).

## Attribution and license

The code is MIT (see `LICENSE`). The default model is **pyannote/speaker-diarization-community-1 by pyannoteAI, licensed [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/)** (<https://huggingface.co/pyannote/speaker-diarization-community-1>), run with [pyannote.audio](https://github.com/pyannote/pyannote-audio) (MIT). Every turns document and every transcript written by `assign` carries this attribution, the model id and the snapshot revision in its provenance. Keep the attribution with anything you publish from the results.
