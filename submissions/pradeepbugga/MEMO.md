# MIB Doc Challenge — Technical Memo

**Author:** pradeepbugga · **Solution repo:** https://github.com/pradeepbugga/mib-doc-challenge

## Approach

The system is a deterministic, offline pipeline: no LLM, no trained model, no
network. Roughly 12k lines of Python over PyMuPDF, OpenCV, and Tesseract.

```
predict.py → packet_pipeline → per-page: quality → OCR → classify → extract
           → identity resolution → corroboration → derivation → adjudication
```

The design principle throughout is **evidence precedence over transcription
accuracy**. The `FIELD_MANUAL` ranks sources — signed adjudicator note, intake
form, biometric slip, sponsor attestation, registry extract, raw text layer —
and every field carries the provenance of where it came from. When two pages
disagree, the corroborator resolves by precedence rather than by recency or
confidence. This matters because classification is worth 80 points and
extraction 50: a system that transcribes well but trusts the wrong page loses
more than one that recovers fewer fields from trusted ones.

Adjudication is an ordered rule ladder, first match wins, mirroring the manual:
signed note → disqualifying risk flag → transit visa → unpaid fee → revoked
sponsor → unresolved fee → review-only flag → missing sponsor → insufficient
evidence → staleness → approve. Each rule emits its own confidence, calibrated
against its observed hit rate on the training set (e.g. adjudicator notes agreed
with the label 162/162, so they carry 0.95; the revoked-sponsor rule is
inferential and carries 0.76).

## What actually moved the score

Four findings, each measured by A/B run over the full training set:

**Only visible ink is evidence.** Packets carry prompt-injection payloads as
pure-white size-5 text in the native PDF layer, plus fake answer keys outside
the page crop. Filtering white and off-crop spans removed injected text from
843/843 affected pages with no legitimate text lost. The subtler half of the
same problem is in the *image* path: injected ink renders at ~226–255 against a
255 background — invisible to a human, but CLAHE amplifies it into crisp,
readable glyphs. Clamping near-white to white *before* contrast enhancement
(`suppress_faint_ink`, floor chosen by sweep) both defeats that and removes scan
haze. It was the single largest OCR win: classification on degraded pages went
45.4% → 63.8%.

**Damage destroys labels more than values.** A value sits inside one scan band;
a longer label straddles a band boundary. So the extractors match labels
fuzzily (0.78 similarity, 0.10 margin) and recover values three ways —
controlled-vocabulary n-gram search, fixed patterns for structured IDs, and
noise-cut free text. Biggest single extraction win of the project.

**Repairs must be scored, never applied unconditionally.** Orientation
correction, scan-band tear repair, and text-line realignment are all *candidates*
that are adopted only if they beat the current page result under a scoring
function. Applying tear repair unconditionally measured −31 fields; as a scored
retry it is neutral-to-positive. This is why no single repair can regress a page.

**The biggest wins were downstream of OCR, not in it.** Two bugs in series each
silently discarded whole pages: a page whose footer failed OCR got no `case_id`
and was dropped entirely (16% of pages, 38% of packets losing at least one
correctly-extracted value), and identity conflicts were compared *exactly*, so
OCR damage manufactured them — `'Orirx Orivoss Spr~- "te.'` vs `'Oririx
Orivoss'` is 0.963 similar but read as a different applicant. Fixing either
alone showed almost nothing; together they were the largest jump of the project.
Structured IDs still compare exactly, because `SPN-1680` vs `SPN-1690` is 87%
similar and a genuinely different sponsor.

## Results (training set, 1,000 packets)

| Section | Score |
| --- | --- |
| Field extraction | 39.68 / 50 |
| Classification | 60.91 / 80 |
| Calibration | 12.45 / 20 |
| Missing-case penalty | −0.00 / 10 (0 missing cases) |
| **Total** | **113.04 / 150** |

Mean confidence Brier 0.1889; 43 catastrophic false approvals.

Per-field accuracy:

| Field | Accuracy | Field | Accuracy |
| --- | --- | --- | --- |
| `species_code` | 91.5% | `arrival_date` | 80.5% |
| `declared_purpose` | 89.6% | `risk_flags` | 72.1% |
| `home_world` | 88.0% | `applicant_name` | 67.5% |
| `visa_class` | 84.9% | `fee_status` | 59.9% |
| `sponsor_id` | 82.6% | | |

The dominant classification error is caution rather than recklessness — 78
APPROVED and 64 DENIED cases were routed to NEEDS_REVIEW, which the scorer pays
partial credit for, against 43 DENIED cases wrongly approved.

## Failure modes

**A hard ceiling in the rules, not the OCR.** Running the adjudication engine on
*ground-truth* field values scores only 88.6% — so even with perfect extraction
the system would reach about 136/150. The residual is roughly 111 cases of pure
rule error, dominated by: clean-looking packets whose label is DENIED (37) or
NEEDS_REVIEW (27); the manual's "multiple review-only flags may combine into a
denial in edge cases", which I never implemented; and false positives from
inferred revoked sponsors (21), which is unavoidable given those sponsors were
recovered at 67–83% denial rates.

**Calibration is not an independent lever.** I repeatedly mis-read this as free
points. Brier score is bounded by accuracy — at 70% accuracy a *perfectly*
calibrated system scores 11.62/20, and the system was already at 11.28. Only
~0.34 points were ever available. Calibration rises only when accuracy rises.

**Some cases are irreducible.** A meaningful fraction of flagged packets carry a
disqualifying flag in the label with zero visible evidence anywhere in that
packet — verified exhaustively for several cases across drawings, images,
annotations, and OCG layers. These read as deliberate "you cannot know this"
examples. `risk_flags` (72.1%) needs three separate mechanisms: literal text on
a biometric slip, a `registry_status: EMBARGO REVIEW` mapping, and this
unrecoverable remainder.

**Weakest fields:** `fee_status` and `applicant_name`. The fee derivation
deliberately *abstains* when the receipt shows amount 0 with no waiver code,
because that case splits evenly between unpaid and unknown and guessing "unpaid"
drives a DENIED decision on a coin toss — a catastrophic-false-approval
trade I chose not to make.

## With another week

1. **Implement the combined-review-flag denial rule.** It is stated in the
   manual, worth ~26 cases of pure rule error, and I never got to it.
2. **Attack the 37 clean-packet-but-DENIED cases.** These are the catastrophic
   false approvals and the largest single scoring liability.
3. **Replace inferred revoked sponsors with a calibrated prior** rather than a
   hard set, so a 70%-denial sponsor produces low confidence rather than a
   confident wrong DENIED.
4. **Cross-document derivation of `identity_conflict` and `sponsor_mismatch`.**
   The pipeline already resolves identity across pages and knows when values
   disagree; it just never writes that back out as a risk flag.
5. **Re-test rendering DPI.** An early measurement said 300 DPI was worse, but
   that predates faint-ink suppression, and spot checks now show 300 DPI
   recovering text that 200 DPI turns to noise.
