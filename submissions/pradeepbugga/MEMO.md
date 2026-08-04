<!--
TEMPLATE NOTES (delete this comment block before submitting):

This is a working draft, not a finished memo. Everything in it is a real,
measured fact as of tonight's session (train set, 1000 packets) — I pulled
every number directly from scripts/evaluate.py output or from live queries
against the predictions/labels, nothing is invented. But the prose is mine
(Claude's) trying to sound like you, not actually you. Places marked
[FILL IN] need your voice/judgment specifically. Everything else you should
still read critically and edit — cut what doesn't feel true, reweight what
matters most to you, add color from earlier sessions I don't have full
context on.

Known gap: the "Results" table below is the TRAIN set (1000 packets), since
that's what's measurable right now. Replace it with the real validation-set
numbers once scripts/evaluate.py has run against the actual submitted
predictions.jsonl (5,000 packets) — the two will be close but not identical,
and the submission should report the real one.
-->

# MIB Doc Challenge — Technical Memo

**Author:** pradeepbugga · **Solution repo:** https://github.com/pradeepbugga/mib-doc-challenge

## Approach

The system is a deterministic, offline pipeline: no LLM, no trained model, no
network. [FILL IN: rough line count if you want it — `wc -l $(find core scripts -iname "*.py")` gives ~13.2k across core/ + scripts/, ~7-8k in core/ alone] of Python over PyMuPDF, OpenCV, and Tesseract.

```
predict.py → packet_pipeline → per-page: quality → OCR → classify → extract
           → corroboration → derivation → adjudication
```

[FILL IN / VERIFY: the pipeline diagram above drops "identity resolution" from
the old memo's version — that whole subsystem (case-id detection/scoring)
was deleted this session after measurement showed trusting the filename case_id
directly, with zero fallback resolution logic, scored the same or better with
far less code and zero genuine multi-applicant contamination found across
1000 packets. If this doesn't match your mental model of what shipped, fix it.]

The design principle throughout is **evidence precedence over transcription
accuracy**. The `FIELD_MANUAL` ranks sources — signed adjudicator note, intake
form, biometric slip, sponsor attestation, registry extract, raw text layer —
and every field carries the provenance of where it came from. When two pages
disagree, the corroborator resolves by precedence rather than by recency or
confidence, with narrow, measured exceptions where precedence was empirically
backwards for one specific field (see below). This matters because
classification is worth 80 points and extraction 50: a system that transcribes
well but trusts the wrong page loses more than one that recovers fewer fields
from trusted ones.

Adjudication is an ordered rule ladder, first match wins, mirroring the manual.
[FILL IN: confirm the current rule order in core/adjudication/engine.py is
still accurately summarized by the old memo's "signed note → disqualifying
risk flag → transit visa → unpaid fee → revoked sponsor → unresolved fee →
review-only flag → missing sponsor → insufficient evidence → staleness →
approve" — staleness was revived this session after being found as dead code,
so its position in the ladder is worth double-checking.]

## What actually moved the score

[FILL IN: this section is the highest-value part of the memo and the part
that most needs your own judgment about what to keep, cut, or reweight — I've
listed the findings from tonight's session in rough order of measured impact,
but you have context from earlier sessions I don't. Pick the ones that
actually tell the story of how this system got built, not just the ones from
tonight.]

Findings from tonight's session, each measured by a full-training-set A/B run:

**Removing the case-id resolution subsystem was a net win, not a compromise.**
The pipeline used to detect and score candidate case IDs per page to resolve
which packet each page belonged to. Measured directly: zero genuine
multi-applicant contamination in 1000 training packets, and trusting the
filename-derived case_id directly scored *better* (+0.57) with several hundred
fewer lines of code. Sometimes the fix for a fragile subsystem is deleting it.

**A validated-but-unshipped repair technique needed one more piece than its
own writeup said.** A scanline-tear repair prototype (per-row mark detection,
outlier-band cleanup, cascade repair anchored at the page's most typical band)
had been validated on 2 pages in a prior session but never ported into the
real pipeline. Porting it directly gave a small net *regression* on the full
training set — tracing it down, the old implementation's text-corroboration
check (only trust a tear boundary if body text also shifted, not just the
tracked margin mark) had been dropped in the port, and restoring it (with a
looser "reject only on active disagreement, not absence of evidence" policy,
since the new technique produces far more candidate boundaries than the old
one ever evaluated) flipped it to a genuine improvement (+0.04, one fewer
catastrophic false approval).

**Catastrophic false approvals are dominated by one root cause, and it isn't
a bug.** Investigated all 27 DENIED-predicted-APPROVED cases directly against
the source PDFs: 23 have the disqualifying risk flag's only possible evidence
source (a biometric_slip page) genuinely absent from the packet — not
misclassified, not OCR-failed, structurally not there. [FILL IN: your own
framing here — I'd call this out plainly rather than try to engineer around
it, since EVALUATION.md itself acknowledges fields can be "genuinely
unrecoverable because visible evidence was cut out" for extraction scoring;
the classification penalty doesn't extend that same leniency, which is worth
naming as a real tension in the task, not a gap in this solution.] One of the
27 was a real, fixable miss: the flag's own OCR read as "bichaxerd_yed" for
"biohazard_red" (0.69 similarity, just under the fuzzy-match floor) —
loosening that floor by a validated, false-positive-free margin recovered it.

[FILL IN: earlier-session findings worth keeping from the old memo if they're
still accurate — the "only visible ink is evidence" prompt-injection defense,
"damage destroys labels more than values" extraction strategy, and "repairs
must be scored, never applied unconditionally" safety pattern were all real,
specific, measured findings. I'd keep whichever of these still ring true
rather than have me guess at their current numbers.]

## Results (training set, 1,000 packets)

<!-- [FILL IN: replace with validation-set numbers once available] -->

| Section | Score |
| --- | --- |
| Field extraction | 41.81 / 50 |
| Classification | 65.17 / 80 |
| Calibration | 13.60 / 20 |
| Missing-case penalty | −0.00 / 10 (0 missing cases) |
| **Total** | **120.58 / 150** |

Mean confidence Brier 0.1599; 26 catastrophic false approvals.

Per-field accuracy:

| Field | Accuracy | Field | Accuracy |
| --- | --- | --- | --- |
| `species_code` | 94.0% | `arrival_date` | 83.6% |
| `declared_purpose` | 91.5% | `applicant_name` | 81.4% |
| `home_world` | 90.4% | `risk_flags` | 76.6% |
| `visa_class` | 87.6% | `fee_status` | 62.7% |
| `sponsor_id` | 85.8% | | |

The dominant classification error is caution rather than recklessness — 79
APPROVED and 54 DENIED cases were routed to NEEDS_REVIEW, which the scorer
pays partial credit for, against 26 DENIED cases wrongly approved.

## Failure modes

**A hard ceiling in the rules, not the OCR.** Running the adjudication engine
on *ground-truth* field values scores 143.43/150 (extraction 50/50,
classification 75.95/80, calibration 17.48/20) — so even with perfect
extraction the rule set itself caps out well short of 150. [FILL IN: the
residual rule-error cases were characterized this session as three buckets —
27 clean-looking packets labeled NEEDS_REVIEW with no field-level pattern
found, 18 packets where the manual's "multiple review-only flags may combine
into denial" hint was tested directly and only explains 5 of them, and 9
packets that look like irreducible label noise at n=1 sample sizes. Decide
how much of this granularity is worth including vs. just stating the ceiling.]

**Calibration is not an independent lever.** [FILL IN / VERIFY: this framing
from the old memo — Brier score being bounded by accuracy, so calibration
rises only when accuracy rises — is a real mathematical property and probably
still worth keeping, but the specific numbers (11.62/20 ceiling at 70%
accuracy) should be recomputed against current accuracy if you keep this.]

**Some cases are irreducible by design, not by extraction quality.**
`risk_flags` (76.6%) needs three separate mechanisms to reach its ceiling:
literal text on a biometric_slip page, a `registry_status: EMBARGO REVIEW`
mapping for packets with a registry_extract page but no biometric_slip, and an
unrecoverable remainder where neither document is present. Investigated
exhaustively this session across the 27-case catastrophic-false-approval set:
the registry_status fallback doesn't help there either (the one case with a
registry_extract page reads `CLEAR` despite a true `planetary_embargo` label —
`CLEAR` is a known-ambiguous value, not a reliable negative signal).

**Weakest fields remain `fee_status` and `risk_flags`.** [FILL IN: the old
memo's note about fee_status deliberately abstaining on amount-0/no-waiver
cases to avoid a coin-flip catastrophic-approval trade — worth verifying this
logic is still in place in core/adjudication/fee_derivation.py and still the
right call before restating it.]

## With another week

[FILL IN: prioritize based on what you actually believe is highest-leverage.
Rough status of the old memo's list, for reference —]

1. ~~Combined-review-flag denial rule~~ — [VERIFY: not confirmed done or not
   this session; check core/adjudication/engine.py before claiming either way]
2. Attack the clean-packet-but-DENIED cases — partially addressed
   (catastrophic false approvals 43 → 26 across the project), but the
   remaining 23 are structurally unrecoverable (see above), not a to-do.
3. Replace inferred revoked sponsors with a calibrated prior — [VERIFY status]
4. ~~Cross-document derivation of `identity_conflict`/`sponsor_mismatch`~~ —
   shipped: recall on these two flags went from 33%/41% to 69%/50%.
5. Re-test rendering DPI — [VERIFY status]
6. [FILL IN: the actual highest-leverage remaining item from tonight's
   investigation was probably the extraction-gap arithmetic — roughly 19.7
   points are reachable through better extraction alone before hitting the
   143.43 rule-set ceiling, concentrated in fee_status and risk_flags.]
