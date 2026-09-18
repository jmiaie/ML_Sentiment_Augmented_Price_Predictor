# CITATION-RED-TEAM — D10-D (sentiment-study)

Adversarial review of provenance: every path, hash and mapping in this pack. The
specific failure this document exists to hunt is a hash whose tail was written by
hand — a value that looks measured but was not.

Disposition vocabulary: **UPHELD**, **REPAIRED**, **REJECTED**.

---

**CIT-1 — "Are the artifact hashes measured or typed?"**
Measured. All eighteen declared source-artifact hashes were recomputed from disk
during authoring and compared against the values declared in
`reproducibility.json`: eighteen files, zero corrections required.
`publication_pack.py check` re-hashes every declared artifact and every cited path
on every run and fails closed on any mismatch. Disposition: **REJECTED** as a
defect, and mechanically enforced thereafter.

**CIT-2 — "Are the generated-file hashes in the citation table measured?"**
They are now. Four generated-file hashes were initially written by hand — a true
instance of the failure mode this document hunts — and were caught and replaced
with measured values. `hashcheck` now rejects any published hash that is not a
declared source artifact, a generated file, or a manifest-anchored canonical value,
so a hand-written hash cannot survive the gate. Disposition: **REPAIRED**.

**CIT-3 — "Two dataset hashes cannot be recomputed from this repository. Are they
being asserted on faith?"**
They cannot be recomputed, because raw filing text and price CSVs are not committed
to this repository. They are not asserted on faith either: the exact canonical
string must be found physically present in a named, hash-verified anchor file, and
`hashcheck` fails if it is not. Each canonical value is anchored in its own frozen
manifest and in both 2025 artifacts independently, so a fabricated value would have
to already exist inside three hash-verified files including two result artifacts.
Disposition: **UPHELD as a real limitation** (the value is manifest-declared, not
byte-recomputable here), **REJECTED as an unanchored claim**, and disclosed in
`SOURCE-GATE.md` Field 6.

**CIT-4 — "Do the citation rows resolve to the files the claims name?"**
Every row is parsed by `check`, which re-hashes the cited path and compares it to
the declared value. A row whose path and hash disagree fails the build. The
mapping from claim to field is given in `RESULT-SOURCE-MAP.md`'s "Numbers and where
they come from" table so a reviewer can re-derive a number rather than trust a
citation. Disposition: **REJECTED** as a defect.

**CIT-5 — "Is any important source cited by name but never hashed?"**
`research/historical-textual-signal-study.md` is named by the frozen configuration
as the required report and is absent at the accepted head. The pack cites the
requirement, records the absence, and sources no prose from it. No other named
source appears in prose without a hash or an explicit absence statement.
Disposition: **UPHELD as a known limitation**, disclosed in `SOURCE-GATE.md`
Field 13 and `TECHNICAL-PAPER.md` §9.6.

**CIT-6 — "Do the v1 artifacts appear in any table or figure?"**
No. Three superseded v1 result artifacts exist in the tree and are hashed in the
citation table solely to record that they are preserved and excluded. No generated
table, figure, metric or claim draws on them. `period-metrics.csv` contains no v1
content. Disposition: **REJECTED** as a spillover risk.

**CIT-7 — "Does the producing-code head cited by the pack match what the artifacts
record?"**
Yes, and the two heads legitimately differ by period: the 2025 artifacts record
`0fa6cfd3e124f36eea3383af7da2f71b6c43934f`, the pre-2025 artifacts record
`e55cb09e8367053c24270743e8b8d96f5fe1e375`. This is stated in `SOURCE-GATE.md`
Field 7 and `TECHNICAL-PAPER.md` §11 so the difference is not mistaken for drift,
and no artifact was rewritten to make the heads agree. Disposition: **REJECTED** as
a defect; disclosed as expected provenance.

**CIT-8 — "Can the external git references be resolved from this tree?"**
Two external git SHAs are declared and are not locally resolvable; `hashcheck`
reports them as declared-but-not-locally-resolvable rather than silently passing
them. The alternative — omitting them — would make the pack's provenance look
tighter than it is. Disposition: **UPHELD as a deliberate transparency choice**.

**CIT-9 — "Does the generated table of measured-versus-declared artifact hashes
depend on the generator being right about the generator?"**
Partly. `tables/artifact-hashes.csv` is produced by the same script that `check`
runs, so a script that mis-hashed everything consistently would agree with itself.
Two things bound that risk: `check` compares declared values against files re-read
from disk, and the citation table in `RESULT-SOURCE-MAP.md` duplicates sixteen of
those hashes as literal text inside a document, where a reviewer and CI can
compare them independently of any script. Disposition: **UPHELD as a residual
coupling**, mitigated by the duplicated literal values.

**CIT-10 — "Does the CJK/typography in these documents hide character
corruption?"**
Ellipses and dashes are used consistently, and no prose depends on a glyph for
meaning. Disposition: **REJECTED**.

## Summary

One defect was found and repaired during authoring (CIT-2, hand-written generated
hashes); one limitation is upheld and disclosed by design (CIT-3, manifest-declared
canonical dataset hashes); one coupling is upheld with mitigation (CIT-9). No
citation in this pack resolves to a missing file, and no hash in it is unanchored.

## Self-caught defects in this pack (audit trail)

These are recorded rather than erased. Each was found by a gate in this pack, not
by a reviewer, and each is reproducible from the commit history.

1. **Four generated-file hashes were typed by hand.** The citation table's rows for
   generated outputs initially carried hex transcribed from memory rather than read
   from disk. This is the same defect class as a fabricated hash tail. All four were
   replaced with measured values, and `hashcheck` now rejects any hash that is not
   anchored to a named, hash-verified file.
2. **The hash-anchoring check used its failure list before initialising it.** The
   bug was invisible because the failure branch had never been taken. It surfaced
   only when the gate was deliberately negative-tested. Fixed; the negative test now
   confirms the gate fails closed and reports cleanly.
3. **A development-block delta was written at the wrong precision.** The paper
   stated `+0.0202998`; the measured value is `+0.02029985`. The cross-checker's
   prose assertion caught it, and the section table and abstract were regenerated
   from the artifact.
4. **The cross-checker read a key from the wrong nesting level.** It looked for
   `max_train_target_end_ordinal` inside the training-geometry block and raised a
   `KeyError` on its first complete run. The assertion had never passed silently; it
   had never been reached.
5. **The cross-checker applied the wrong arithmetic to the corpus counts.** It
   asserted that form counts sum to filing events read. They sum to frame rows,
   2,296, which equal events read, 2,349, minus every documented skip. The paper was
   correct; the checker was corrected instead.
6. **The overclaim guard produced a false positive on a quoted attack.** A red-team
   question quoted the phrase being refuted, and the quotes spanned a line break, so
   no balanced span was recognised and the quoted text was read as an assertion. The
   guard now uses QUOTE-EXEMPT handling and fails closed on malformed quoting. The
   prose was reflowed to keep the attack on one line; the guard was not weakened.
