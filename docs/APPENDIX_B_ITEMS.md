# Appendix B — Item-level disclosures

*Referenced from the project's pre-release checklist ("weakest-items confession" and "threshold-robustness
table"). Written from the released artifact (`battery/battery.json`, `battery/l0_verification.json`,
`battery/l0n_sources.json`) rather than from memory of the construction process.*

## B.1 Known item-level imperfections

### L1 answer-token asymmetries

L1 scores the first token after a single-fact cloze. Five of the 24 L1 items have a gold answer that
is not one token on both models, which breaks a same-token comparison across the Pythia/OLMo-2
tokenizer boundary for that item:

| Item | Answer | Pythia tokens | OLMo-2 tokens | Problem |
|---|---|---|---|---|
| `L1-weather-2` | thermometer | 2 (` therm`+`ometer`) | 1 | A first-token metric on Pythia scores the prefix ` therm`, not the full word. |
| `L1-Greek_mythology-1` | Zeus | 2 (` Ze`+`us`) | 1 | ` Ze` is at least a specific prefix, so this one is tolerable with span scoring. |
| `L1-Greek_mythology-2` | Poseidon | 4 (` P`+`ose`+`id`+`on`) | 2 (` Pose`+`idon`) | Worst case: Pythia's first token ` P` is shared with thousands of words, so a first-token metric measures almost nothing for this item on Pythia. |
| `L1-cooking-1` | 100 | 1 | 2 (` `+`100`) | OLMo-2 splits the leading-space digit string, so the token immediately after the cloze is a bare space carrying zero answer information. |
| `L1-chess-1` | 64 | 1 | 2 (` `+`64`) | Same digit-splitting issue; the corpus-dominant continuation ("a chessboard has 64 squares") is exactly the form OLMo-2 fragments. |

All five are scored correctly by full-span (teacher-forced) comparison; the defect is specific to a
bare first-token metric, and `L1-Greek_mythology-2` is the one item where that metric should not be
trusted on Pythia at all.

### Answer-set ambiguity, `L1-classical_music-1`

The item asks which orchestral family the violin, viola and cello belong to. The scored gold
continuation is `string`, but `strings` (plural, as in "the string section" vs. "the strings") is an
equally natural completion and a distinct single token on both models. This item's answer set should
be read as `{string, strings}`, not `string` alone — the same class of surface-form smear already
flagged for `L1-gardening-2` (roots/root), `L1-mathematics-2` (pi/Pi), `L1-human_anatomy-2`
(skin/liver), and `L1-cooking-2` (eggs/egg) in `battery.json`'s own `qc_problem` fields.

### Gutenberg wording variant, `L0-12` (Gettysburg Address, PG4)

The L0 item's prompt and gold continuation are drawn verbatim from Project Gutenberg's digitization
of the Gettysburg Address (`gutenberg.org/ebooks/4`), not from the wording most commonly reproduced
today. Compared with the Bliss text (the version engraved on the Lincoln Memorial and reprinted in
most anthologies and textbooks), the Gutenberg edition:

- reads "brought forth **upon** this continent" where the Bliss text reads "brought forth **on** this
  continent" (the Gutenberg wording follows an earlier draft reading, not the final one);
- uses a colon and lowercase "a new nation**:** conceived in **liberty**" where the Bliss text uses a
  comma and capitalizes "a new nation**,** conceived in **Liberty**";
- renders two clause boundaries as ". . ." ellipses ("great civil war. . .testing", "so
  dedicated. . . can long endure") that do not appear in the standard quoted text.

A model that has memorized the version of the address most people have actually read — the Bliss
text — will not get credit against this item's gold string at exactly these three points, even
though it plainly "knows" the passage. This is a property of which historical printing Project
Gutenberg digitized, not a construction error; it is disclosed here rather than silently affecting
the L0 memorization scores.

### L0N control sources: one reduced-eligibility item, one non-English item

Of the twelve L0N (matched non-memorized control) sources, corpus verification against the OLMo
pretraining mix found a non-zero match for PG67345 (max probe count 5, against 0 for the other eleven
controls on that corpus). PG67345 is therefore eligible for Pythia-1.4B only (`battery.json`
`L0N-10`) and is not scored against OLMo-2-1B; it was not dropped from the released battery. Full
per-corpus counts are in `battery/l0n_sources.json`.

Separately, PG64502 is a Hungarian-language work ("Életemből", Mór Jókai), unlike the other eleven
L0N controls and all fourteen L0 sources, which are English. It was kept because the L0N role only
requires a genuinely non-memorized passage under both corpora, not English prose specifically, and
its near-zero probe counts hold regardless of language. It is disclosed here because a reader
skimming the battery for "obscure Gutenberg books" would not expect one control passage to be in a
different language from the rest of the study.

## B.2 L0 memorization-gate threshold scan

L0 items are treated as "corpus-verified memorized" if a large enough fraction of their overlapping
11-word probe windows return a corpus match count at or above a threshold. `battery/l0_verification.json`
stores, for each of the 20 L0 items and each corpus (`pile`, `olmo_mix`), the raw match count for
every probe window. This table sweeps both parameters of that gate — the count threshold and the
required window-fraction — and reports how many of the 20 L0 items pass, per corpus, at each
combination. Passing is defined as: (probes with `count ≥ threshold`) / (probes with `status == "ok"`)
≥ window-fraction.

| window-fraction | count ≥ 10 (pile / olmo) | count ≥ 20 (pile / olmo) | count ≥ 40 (pile / olmo) |
|---|---|---|---|
| 0.6 | 18 / 20 | 17 / 20 | 16 / 20 |
| 0.7 | 18 / 20 | **17 / 20** | 15 / 19 |
| 0.8 | 17 / 20 | 13 / 19 | 13 / 18 |
| 0.9 | 15 / 19 | 12 / 19 | 11 / 16 |

The rule actually used in the pilot (window-fraction 0.7, count ≥ 20 — the "70%/20" rule referenced
in the pre-release checklist) passes **17/20 items on the pile corpus and 20/20 on the OLMo mix**
(bolded cell above). The gate is not brittle to nearby settings: relaxing to 0.6 or tightening the
count to 40 moves the pile pass count by at most 2 items in this sweep, and the OLMo mix stays at or
above 19/20 across the whole 0.6–0.8 × 10–40 region. It only degrades meaningfully at the strictest
corner (window-fraction 0.9, count ≥ 40: 11/20 pile).

## B.3 L0N control sources

All twelve L0N control passages, mapped from the machine identifiers recorded at selection time
(`battery/l0_control_candidates.json`) to their real titles and authors via the Gutendex API
(`gutendex.com/books/<id>`). Full mapping, including per-corpus verification counts, is in
`battery/l0n_sources.json`.

| Gutenberg ID | Title | Author | Eligibility |
|---|---|---|---|
| PG62012 | A Week in Wall Street: By One Who Knows | Frederick Jackson | Pythia-1.4B + OLMo-2-1B |
| PG62345 | Collectors' Items: Fifty Superb Recipes from Spice Islands | Spice Islands Company | Pythia-1.4B + OLMo-2-1B |
| PG63111 | The Grenadier Guards in the Great War of 1914-1918, Vol. 3 of 3 | Frederick Ponsonby (1867–1935) | Pythia-1.4B + OLMo-2-1B |
| PG63888 | The Great American Novel | William Carlos Williams (1883–1963) | Pythia-1.4B + OLMo-2-1B |
| PG64502 | Életemből (II. rész): Igaz történetek. Örök emlékek. Humor. Utleirás. | Mór Jókai (1825–1904) | Pythia-1.4B + OLMo-2-1B (Hungarian-language; see B.1) |
| PG65123 | Gold Hunting in Alaska | Joseph Grinnell (1877–1939) | Pythia-1.4B + OLMo-2-1B |
| PG65789 | Sweaters He and She | American Thread Company | Pythia-1.4B + OLMo-2-1B |
| PG66234 | The Abergeldie Winter Book | Eléonore Riego de la Branchardière | Pythia-1.4B + OLMo-2-1B |
| PG66890 | A Square Deal | Theodore Roosevelt (1858–1919) | Pythia-1.4B + OLMo-2-1B |
| PG67345 | The Wonderful Adventures of Phra the Phoenician | Edwin Lester Arnold (1857–1935) | Pythia-1.4B only; see B.1 |
| PG67901 | Frank Merriwell in Europe; or, Working His Way Upward | Burt L. Standish (1866–1945) | Pythia-1.4B + OLMo-2-1B |
| PG69012 | Drawing in charcoal and crayon for the use of students and schools | Frank Fowler (1852–1910) | Pythia-1.4B + OLMo-2-1B |

All twelve works are unambiguously in the US public domain: federal copyright term expired, and
Project Gutenberg only distributes texts it has cleared for US public-domain status (confirmed
directly via each work's Gutendex `copyright: false` field). No claim of copyright is made or
implied over the excerpted source text itself — see `LICENSE-DATA.md` §2 for the project's rights
statement covering both the L0 and L0N excerpts.
