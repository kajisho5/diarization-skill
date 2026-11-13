# Speaker-id assignment rule

This repository owns the rule that turns speaker turns into `speaker_id` values in a transcription-skill transcript. `diarization assign` implements it (`src/diarization_skill/assign.py`, method `max_overlap_with_exclusive_turns`). transcription-skill's speaker merge is a copy of the same rule. Both run the conformance fixtures in [`conformance/speaker_id/`](../conformance/speaker_id/) (ADR-004).

## Inputs

- A transcript, `transcription-skill/transcript/0.1`: segments with `start`, `end` and `words` (a list of timed words, or `null`).
- A turns document, `diarization-skill/turns@1`. Only its exclusive `turns` (one speaker at a time, ordered by start) are used. The overlap block (`overlap.turns`, `overlap.regions`) never adds or changes a speaker.

## Rule

1. **Overlap.** The overlap of a unit `[s, e]` with a turn `[ts, te]` is `min(e, te) - max(s, ts)` when it is positive, else zero. A unit that only touches a turn edge (`e == ts` or `s == te`) has zero overlap with it. For each speaker, the overlaps with all of that speaker's turns are summed.
2. **Pick.** The unit gets the speaker with the strictly largest summed overlap. It stays `null` when every overlap is zero, or when two speakers share the largest value. Two values are equal when they differ by at most `1e-9` s.
3. **Words.** In a segment with a non-empty `words` list, every word is a unit and gets its own `speaker_id` (rules 1-2).
4. **Segment with words.** The segment's speaker is picked (rule 2) from votes: each word that got a speaker votes with its duration `end - start` for that speaker. Words left `null` do not vote. A tie, or no assigned word, leaves the segment `null`.
5. **Segment without words.** A segment with `words: null` (or an empty list) is itself the unit (rules 1-2).
6. **Nothing is guessed.** No nearest-turn fill, no carry-over from neighbouring words or segments, no smoothing. Ids, text and times of the transcript are never changed.
7. **Overlap report.** A segment is listed in `speaker_assignment.overlap_segment_ids` when it intersects an `overlap.regions` entry (`region.start < segment.end` and `region.end > segment.start`). This is a report only; it does not change any `speaker_id`.

## Conformance fixtures

`conformance/speaker_id/manifest.json` (`diarization-skill/speaker-id-conformance@1`) lists the cases with the SHA-256 of each case file and the `fixture_version`. Each case (`diarization-skill/speaker-id-case@1`) holds a full transcript, a full turns document and `expected`:

- `segments`: `[{id, speaker_id, words}]`, where `words` is the expected `speaker_id` of each word in order, or `null` for a segment without words;
- `overlap_segment_ids`: the expected list, in segment order.

The cases cover a word inside a turn, a word across a turn boundary, a gap with no turn, a word that only touches a turn edge, ties (word, segment, segment votes), overlapping speech, null words that do not vote, and overlap summed over several turns of one speaker. All times are multiples of 0.25 s, so every comparison is exact in binary floating point.

An implementation conforms when, for every case, the `speaker_id` of every segment and word and the overlap segment list equal `expected`. A consumer copies the directory as it is and pins the `fixture_version`. Any change to a case file, the manifest or the expected values bumps `fixture_version`; this repository's tests fail if a case changes without that bump.
