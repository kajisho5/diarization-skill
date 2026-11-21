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

## ADR-004 The speaker-id rule as a spec with shared conformance fixtures

Roadmap step 9. Decided 2026-11-13.

### Context

The Velisa ownership decisions of 2026-11-12 give the speaker-id assignment rule (turns -> `speaker_id`) to this repository. transcription-skill's speaker merge (its ADR-037) is a copy of ASSIGN's rule, so that the agent can merge speakers without installing this skill. Two copies drift unless both are checked against the same cases.

### Decision

- **The rule is written down** in [docs/speaker_id_rule.md](speaker_id_rule.md): overlap with exclusive turns only, summed per speaker; strictly largest wins, zero or a tie (within `1e-9` s) is `null`; words vote for their segment by duration, `null` words do not vote; no inference from neighbours; the overlap regions only produce the `overlap_segment_ids` report. It is what `assign.py` already did. No behaviour changes.
- **Fixtures are data in the repository**, in `conformance/speaker_id/`: a manifest (`diarization-skill/speaker-id-conformance@1`) with `fixture_version` and the SHA-256 of each case, and one file per case (`diarization-skill/speaker-id-case@1`) with a full transcript, a full `diarization-skill/turns@1` document and the expected `speaker_id` per segment and word plus `overlap_segment_ids`. Six cases cover a word inside a turn, a word across a turn boundary, a gap with no turn, a word that only touches a turn edge, ties (word, segment, segment votes), overlapping speech, null words that do not vote, and overlap summed over several turns of one speaker.
- **Exact by construction.** Every time is a multiple of 0.25 s, so the expected values do not depend on floating-point rounding, and the inputs are the full documents both validators accept.
- **Versioned and pinned.** `tests/test_speaker_id_conformance.py` pins the manifest's SHA-256 per `fixture_version` and checks every case's hash. Changing a case without a new `fixture_version` fails here. `.gitattributes` keeps the fixture bytes identical on Windows checkouts.
- **Consumers copy, then pin.** A consumer vendors the directory unchanged and pins the same `fixture_version` and manifest hash. Comparing its copy with a checkout of this repository in its CI makes a fixture change here fail there until it is taken. transcription-skill does this (its roadmap step 17).
- **CI.** ASSIGN runs every case in the unit matrix. The integration job, which already checks out transcription-skill, also runs `transcription speakers` on every case, so the copy is checked from this side too.
- **No contract change.** The fixtures are not a tool; `diarization contract` is unchanged.

### Consequences

- A change to the rule is a change to the fixtures: new `fixture_version`, new pin, and the consumer has to take it before its CI is green again.
- The fixtures fix the `speaker_id` values and `overlap_segment_ids`, not the warnings text or the `speaker_assignment` provenance block, which differ by design (tool, skill).

## ADR-005 The contract in the OS contract shape: `skill` as an object, Skill-only `dependencies`

2026-11-13.

### Context

Velisa's contract-shape check (`si contract-check`, Velisa ADR-018) validates every pinned contract against the OS schema, `registry/capability_contract.schema.json`. This contract broke it in two places:

- `skill` was the package name as a string; the schema wants an object (`{id, version}`).
- `dependencies[]` named every entry under `id` and held a Python package, the pipeline weights, a binary and two document schemas; the schema wants Skill dependencies, each with `skill_id` and `version_range`.

The schema stays as it is. The compliant contracts already show the convention: transcription-skill lists only Skills in `dependencies` (none) and leaves its provider packages and programs to `doctor` and `engines`; nle-bridge-skill and publish-skill give `skill` as `{id, package, version}`.

### Decision

- `skill` is `{"id": "diarization", "package": "diarization-skill", "version": <version>}`. `skill_id` and `id` stay.
- `dependencies` is `[]`: this skill invokes no other Skill.
- pyannote.audio, the pipeline weights and ffmpeg are not Skills. `doctor` already reports each with its version and install steps, and `engines` keeps the model id, licence and attribution. The transcript (`assign.input_schema`) and the detection-skill OCR evidence (`naming.evidence`) are documents this skill reads, not Skills it calls; they stay described there.

### Consequences

- `diarization contract --json` passes `si contract-check` against the OS schema with no baseline entry.
- No command, option, output document, tool, capability id or error changed. `contract_version` stays 1.0: by the OS versioning rules (Velisa docs/VERSIONING.md section 3) a bump is for capability shape, and nothing a dependent calls changed. The skill version becomes 0.3.2.
- An orchestrator that compares `dependencies` or `skill` with a pinned copy sees a change and has to re-pin. video-production-agent reads `skill_id`, not `skill` or `dependencies[].id`, and `dependencies` is not one of its fatal drift keys.

## ADR-006 Releases on push to main: the version from pull request labels, the tag and the GitHub Release from one workflow

2026-11-19.

### Context

There was no release workflow and no tag: version 0.3.2 was never released, so dependents pinned this contract from main. The other skills (audio-production-skill, video-editing-skill, color-grading-skill, motion-graphics-skill, qc-skill, subtitle-skill, transcription-skill and more) release from one workflow on push to main, with the version decided by pull request labels.

### Decision

- **`.github/workflows/release.yml` runs on every push to main**, the same job as video-editing-skill's. `.github/scripts/compute_version.sh` decides the version: when `pyproject.toml` equals the latest `v*` tag, release-drafter (dry run, `.github/release-drafter.yml`) resolves the next one from the labels of the pull requests merged since that tag (`feature` / `minor` / `enhancement`: minor; `major` / `breaking`: major; anything else, or no label: patch). Otherwise the `pyproject.toml` version is released as written: the first release, or a version a pull request bumped by hand.
- **The release commit carries the version and the contract.** The job writes the version into `pyproject.toml`, regenerates `contract/contract.json` with `python contract/generate.py`, fails unless `--check` passes and the file reports that version, prepends the commit subjects since the last tag to CHANGELOG.md, commits `chore(release): vX.Y.Z`, pushes an annotated tag and creates the GitHub Release with the same notes. It is one job because a tag pushed with the default token starts no other workflow.
- **Labels.** `.github/workflows/autolabel.yml` labels each pull request from its title, branch and files (`feature`, `fix`, `chore`, `documentation`). `major` / `breaking` is never applied from text, only by a person. A release-level label a bot put on a Dependabot pull request is removed before the version is resolved (`.github/scripts/dependabot_release_labels.py`).
- **Nothing goes to a package index.** The skill is installed from a git checkout.

### Consequences

- The first run (no tag yet) releases `pyproject.toml`'s version unchanged: `v0.3.2`, whose `contract/contract.json` is the one already on main (CI runs `generate.py --check`).
- A pull request no longer bumps the version or writes CHANGELOG.md by hand; it bumps `pyproject.toml` (and regenerates the contract file) only to choose a version the labels would not give. A major version is never chosen without a person's label.
- The tag's `contract/contract.json` is that tag's own contract: `git show vX.Y.Z:contract/contract.json`.
- `tests/test_release_scripts.py` (version decision and file bump, on throwaway repositories) and `tests/test_release_labels.py` (the Dependabot label rule and its wiring) cover the scripts.

## ADR-007 The release job tests its commit before pushing it; chore / ci / docs / test pull requests release nothing

2026-11-19. The ecosystem rule is Velisa ADR-023 (decision D37).

### Context

Today's release commits wrote CHANGELOG.md headings as `## vX.Y.Z - date`. Every hand-written section here reads `## X.Y.Z - date`. No test failed here, but sound-design-skill's main broke on exactly this. The release commit is pushed with the default token, so no workflow ran on it. Also, any merged pull request released a patch, a `ci:` change included.

### Decision

- `.github/scripts/render_changelog.sh` writes `## X.Y.Z - date`. CHANGELOG.md is corrected: renamed `## v0.3.4` and `## v0.3.3`, and removed the `## v0.3.2 - 2026-11-19` section, which repeated the whole history under a second 0.3.2 heading. `tests/test_release_scripts.py` checks the rendered heading and that no heading has a `v`.
- The release job commits and tags locally, runs the unit tests on that commit (step "Test the release commit before pushing it"), and only then pushes. A failure pushes nothing.
- A push to main releases only when something merged since the latest tag is releasable. `.github/scripts/release_gate.py` decides this, and `compute_version.sh` reads its answer as `RELEASABLE`. A pull request with a release-level label releases (`feature` / `minor` / `enhancement`, `fix` / `bug` / `patch`, `major` / `breaking`). So does a pull request with no label, as a patch. A pull request labelled only `chore`, `ci`, `documentation`, `docs`, `test`, `maintenance` or similar releases nothing. This is imagemagick-skill's rule. release-drafter's `default` accepts only major, minor or patch, so the gate runs first, and chore / maintenance / ci / docs no longer map to a patch.

### Consequences

- A version bumped by hand in a pull request is released whatever its labels.
- A release that fails the tests is not made; a later push retries.

## ADR-008 The 1.0 compatibility promise: ASSIGN is `STABLE`; DIARIZE, ENROLL and NAME stay `EXPERIMENTAL`

2026-11-21. Status: Accepted (Velisa ADR-031 / D46; the owner approved the 1.0 transition).

### Context

Velisa ADR-031 sets what 1.0 means for a Skill. From 1.0.0, a change that `si contract-diff` calls breaking, to a part of the contract the promise covers, ships only in a major release. The promise covers:

- the `provides[]` entries marked `STABLE`, and the tools they map to;
- their request, response and error shapes;
- the top-level contract fields that are not tied to a capability.

An `EXPERIMENTAL` entry may still change in a minor release, with a CHANGELOG line.

ADR-031's readiness measurements for this repository, checked again on 2026-11-21:

- **Breaking changes:** none in the tag diffs (v0.3.2 to v0.3.7) or in the `contract:*` labels of the merged pull requests.
- **Open roadmap steps:** only step 6, measuring DER on consented Japanese conference recordings. It is a field test. Whatever it finds can move DIARIZE's and NAME's defaults (`--link-threshold`, `--min-similarity`, `--min-margin`), and both stay `EXPERIMENTAL` below.
- **Known defects:** issue #18 listed two CodeRabbit follow-ups. Both were fixed first, each in its own patch release: rule 2 of `docs/speaker_id_rule.md` now says what the code does (#23), and the conformance test fails instead of skipping when `TRANSCRIPTION_SKILL_DIR` names a wrong checkout (#24). The third item is a nitpick: one more conformance case. Adding a case bumps `fixture_version`, which transcription-skill pins, so it is left for a later minor release.
- **Releases:** ADR-031 counted four (v0.3.2 to v0.3.5; criterion 4 asks for five). The two patch releases above make six, and 1.0.0 is the seventh. ASSIGN's code is older than the tags: `assign.py` has not changed since 0.1.0 (2026-10-30).
- **Agent:** video-production-agent pins this contract (`integration/siblings.lock.json`) and gates it by `SUPPORTED_SKILL_VERSIONS`. It runs DIARIZE as the analysis kind `speaker_turns` (its ADR-086). It merges the turns into its transcripts by this repository's ASSIGN rule (its ADR-096, through transcription-skill's copy of the rule), and its resolver reports `diarization:ASSIGN` from `doctor`. It does not call ENROLL or NAME.

The contract has a `lifecycle` field only on `provides[]` entries. This repository has four: `speech.diarize`, `speech.assign_speakers`, `speech.enroll_voice` and `speech.propose_speaker_names`. `check` and `score` have no entry. A parameter or an output field has no lifecycle field either, so ADR-031 records their status in text: the README, the CHANGELOG and this ADR.

### Decision

#### 1. `STABLE` entry

**`speech.assign_speakers`** (`diarization/assign`). It covers:

- the arguments `<transcript.json> <turns.json> -o <out.json>`, `--replace`, `--allow-source-mismatch`, `--overwrite` and `--json`;
- the rule in [docs/speaker_id_rule.md](speaker_id_rule.md) (method `max_overlap_with_exclusive_turns`) and the conformance fixtures in `conformance/speaker_id/` at `fixture_version` 1. A change to an expected value is a breaking change;
- the input documents: `transcription-skill/transcript/0.1` and the required fields of `diarization-skill/turns@1` that ASSIGN reads (`source.fingerprint`, `turns`, `overlap`, `speakers`, `engine`, `parameters`, `created_at`);
- the output: the same transcript with `speaker_id` per segment and word, and the keys of the `speaker_assignment` block (`skill`, `skill_version`, `tool`, `created_at`, `method`, `rule`, `speaker_ids`, `turns_schema`, `turns_created_at`, `source_fingerprint`, `source_match`, `engine`, `parameters`, `speakers`, `segments_per_speaker`, `counts`, `overlap_available`, `overlap_segment_ids`);
- the error codes ASSIGN raises (`INPUT_NOT_FOUND`, `INVALID_TRANSCRIPT`, `INVALID_TURNS`, `SOURCE_MISMATCH`, `SPEAKERS_ALREADY_SET`, `OUTPUT_IS_INPUT`, `OUTPUT_EXISTS`, `INVALID_ARGUMENT`) and the exit codes.

Reasons:

- It is deterministic, runs locally and needs only the Python standard library. No model, weights or ffmpeg are involved.
- It has shipped since 0.1.0, and `assign.py` is unchanged since then.
- This repository owns the rule (Velisa ownership decisions, 2026-11-12). It is written down as a spec with fixtures that transcription-skill runs too, so it already cannot change without both sides noticing.
- The agent's transcripts carry `speaker_id` by this rule.

A breaking change to anything above, or to the error codes and the top-level contract fields, ships only in a major release (2.0.0).

#### 2. Experimental parts of the `STABLE` tool

These parts of ASSIGN's output stay experimental. Each may still change in a minor release, with a CHANGELOG line:

- **The values copied from the turns document:** `speaker_assignment.engine` and `speaker_assignment.parameters`. ASSIGN copies them as DIARIZE wrote them, so their contents follow DIARIZE and its engine (section 3). The keys of `speaker_assignment` themselves are stable.
- **The text of `speaker_assignment.rule` and of the `warnings` lines.** They are for people. The method id `max_overlap_with_exclusive_turns` is the machine-readable name of the rule.
- **The optional `turns@1` blocks that ASSIGN does not read:** `chunking` (ADR-003) and `speaker_embeddings` (ADR-002). DIARIZE writes them with `--chunk-seconds` and `--embeddings`.

A request with none of these parts in mind gets the `STABLE` behaviour of section 1. They are left out of the promise because they belong to DIARIZE, or they are prose, not because they are known to be wrong.

#### 3. `EXPERIMENTAL` entries

- **`speech.diarize`** (`diarization/diarize`). It runs pyannote.audio and the gated `speaker-diarization-community-1` weights (ADR-001), third-party software and a third-party model this repository does not control. Its turns change with the engine and the model revision. The real engine's DER has not been measured yet (roadmap steps 3 and 6), and no 2-8 h recording has been run through it (step 8). Its newer options stay experimental with it: `--embeddings` (ADR-002), `--chunk-seconds`, `--link-threshold` and `--work-dir` (ADR-003), whose threshold is not calibrated. The agent calls DIARIZE, so a breaking change to it is still named in the pull request and the CHANGELOG, and the agent's re-pin sends it to review (agent ADR-117). The `turns@1` fields that ASSIGN reads are held by section 1.
- **`speech.enroll_voice`** (`diarization/enroll`). It runs the same model to make a voiceprint, so the voiceprint's embedding changes with the model. It was added in 0.2.0 (2026-11-06), and the consent record and the refusal of a sample with other voices have not been reviewed in the field.
- **`speech.propose_speaker_names`** (`diarization/name`). It is deterministic, but it compares embeddings from the model. Its thresholds (`--min-similarity 0.6`, `--min-margin 0.1`) were set on a synthetic fixture and are not calibrated on pyannote's embeddings (roadmap step 7). Calibrating them can change the defaults and the proposals. The agent does not call it.

`check` and `score` have no `provides[]` entry and are not covered by the promise. `check` of a turns document is the same validation ASSIGN runs, so a turns document ASSIGN accepts passes `check`.

#### 4. How the promise is kept

- Promoting an entry (or a part of section 2) to `STABLE` is metadata only. si's `lifecycle-promoted` rule classifies it as `none`, so it can ship in any release. DIARIZE is the next candidate, once roadmap step 6 has measured it.
- **Deprecation** of the `STABLE` entry or one of its options (ADR-031 section 3):
  1. mark it `DEPRECATED` in a minor release, naming its replacement in the README and the CHANGELOG;
  2. keep it working for at least one more minor release and 30 days;
  3. remove it only in the next major release.
- **Breaking-change review.** `major-label-check.yml` fails a `contract:breaking` pull request on 1.x without the `major` label. For a break only in an `EXPERIMENTAL` entry, ADR-031 section 5 lets the reviewer remove `contract:breaking` by hand. si reports an option of a `STABLE` tool as part of that tool. So the reviewer can remove the label only when both of these hold:
  - every breaking item in the contract-diff comment is in `speech.diarize`, `speech.enroll_voice`, `speech.propose_speaker_names` or a part listed in section 2;
  - the pull request and the CHANGELOG entry name it.

#### 5. Release

The 1.0 pull request carries the `major` label. release-drafter then resolves 0.3.x → 1.0.0, and `release_gate.py` treats `major` as release-level. No workflow changes. The contract diff of the pull request is `none` (one `lifecycle-promoted` item). `contract_version` stays `1.0`: it is a separate axis from the release version.

### Consequences

- A caller that uses ASSIGN can float across 1.x minors.
- A caller of DIARIZE, ENROLL or NAME, the agent included, reads the CHANGELOG on each minor release. The agent's re-pin still flags a breaking change in them for review.
- video-production-agent's re-pin sees a 0.x → 1.0.0 release with a `none` diff and widens its gate to `1.0.`.
- `tests/test_diarization.py::ContractTests::test_provides_lifecycle_is_the_1_0_promise` fixes the lifecycles and checks that the `STABLE` tool is deterministic and takes no pipeline.

## ADR-009 The engine's Python floor follows pyannote.audio's own `Requires-Python`

Checked 2026-11-21 against the `requires_python` of each pyannote.audio release on PyPI.

### Context

- pyannote.audio 4.0.0 to 4.0.7 declare `Requires-Python >=3.10`. 4.1.0 (2026-11-17) and 4.1.1 (2026-11-19) declare `>=3.11`.
- The `[pyannote]` extra is `pyannote.audio>=4.0,<5`. pip picks the newest 4.x that fits the interpreter. A `pip download --python-version` resolve gives 4.0.7 on 3.10 and 4.1.1 on 3.11; on 3.9 it finds no 4.x release and fails.
- The comment in `pyproject.toml`, the README, the engine's error and `doctor` said only "Python >= 3.10", which no longer describes 4.1. On Python 3.9 `doctor` offered `pip install 'diarization-skill[pyannote]'` as the step, which cannot resolve there.

### Decision

- The extra stays `pyannote.audio>=4.0,<5` with no environment marker. 3.10 keeps working on 4.0.x. A marker would turn the failing install on 3.9 into a silent install without the engine.
- The comment, the README, the engine's `ENGINE_UNAVAILABLE` hint and `doctor` say: 4.x needs Python >= 3.10; 4.1 and later need >= 3.11.
- On Python < 3.10, `doctor`'s `pyannote_audio` install step is "Python >= 3.10 first, then pip install ...". `doctor`'s readiness logic does not change.

### Consequences

- No contract change, no change to DIARIZE's behaviour. The results' `engine.version` records which pyannote.audio ran, so a 3.10 run (4.0.x) and a 3.11 run (4.1.x) can be told apart.
- `tests/test_diarization.py::PipelineDirTests::test_doctor_python_floor_matches_pyannote_requires_python` fixes the hint per Python version.
