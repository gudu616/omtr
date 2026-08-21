# LICENSE-DATA

## 1 · Produced data and results — CC BY 4.0

The data and results this project produced — the battery items authored for
this study (prompts, scaffolds, verification records in `battery/`), the
pilot measurement outputs in `results/`, and the analysis outputs derived
from them — are licensed under the **Creative Commons Attribution 4.0
International License (CC BY 4.0)**.

> https://creativecommons.org/licenses/by/4.0/

You are free to share and adapt this material for any purpose, including
commercially, provided you give appropriate credit, link to the license, and
indicate if changes were made.

## 2 · Rights statement — third-party public-domain excerpts

The L0 (verbatim-continuation) and L0N (matched non-memorized control)
battery items are not original text of this project. They are short excerpts
drawn from public-domain works distributed by **Project Gutenberg**, all of
which are in the US public domain (federal copyright term expired; Project
Gutenberg only distributes texts it has cleared for US public-domain status).
No claim of copyright is made or implied over the excerpted source text
itself — only the battery's selection, splitting, and verification
methodology around it is covered by §1 above.

### 2.1 L0 sources (`battery/l0_candidates.json`) — 14 works

| Gutenberg ID | Work |
|---|---|
| PG1 | Declaration of Independence |
| PG4 | Gettysburg Address (Lincoln) |
| PG10 | The King James Bible |
| PG11 | Alice's Adventures in Wonderland (Carroll) |
| PG16 | Peter Pan (Barrie) |
| PG36 | The War of the Worlds (Wells) |
| PG76 | Adventures of Huckleberry Finn (Twain) |
| PG84 | Frankenstein (Shelley) |
| PG98 | A Tale of Two Cities (Dickens) |
| PG100 | The Complete Works of William Shakespeare |
| PG120 | Treasure Island (Stevenson) |
| PG1342 | Pride and Prejudice (Austen) |
| PG1661 | The Adventures of Sherlock Holmes (Doyle) |
| PG2701 | Moby-Dick (Melville) |

### 2.2 L0N control sources (`battery/l0_control_candidates.json`) — 12 works

Real titles and authors, looked up from the Gutenberg IDs recorded in the
released artifact via the Gutendex API (`gutendex.com/books/<id>`). Full
mapping, including per-corpus verification counts, is in
`battery/l0n_sources.json`; item-level notes (one non-English source, one
reduced-eligibility source) are in `docs/APPENDIX_B_ITEMS.md` §B.1/B.3.

| Gutenberg ID | Work | Author |
|---|---|---|
| PG62012 | A Week in Wall Street: By One Who Knows | Frederick Jackson |
| PG62345 | Collectors' Items: Fifty Superb Recipes from Spice Islands | Spice Islands Company |
| PG63111 | The Grenadier Guards in the Great War of 1914-1918, Vol. 3 of 3 | Frederick Ponsonby (1867–1935) |
| PG63888 | The Great American Novel | William Carlos Williams (1883–1963) |
| PG64502 | Életemből (II. rész): Igaz történetek. Örök emlékek. Humor. Utleirás. | Mór Jókai (1825–1904) |
| PG65123 | Gold Hunting in Alaska | Joseph Grinnell (1877–1939) |
| PG65789 | Sweaters He and She | American Thread Company |
| PG66234 | The Abergeldie Winter Book | Eléonore Riego de la Branchardière |
| PG66890 | A Square Deal | Theodore Roosevelt (1858–1919) |
| PG67345 | The Wonderful Adventures of Phra the Phoenician | Edwin Lester Arnold (1857–1935) |
| PG67901 | Frank Merriwell in Europe; or, Working His Way Upward | Burt L. Standish (1866–1945) |
| PG69012 | Drawing in charcoal and crayon for the use of students and schools | Frank Fowler (1852–1910) |

All 26 IDs above (14 L0 + 12 L0N) are unambiguously in the US public domain;
this was confirmed by direct inspection during the phase-8 completeness
audit, and independently confirmed here via each work's Gutendex
`copyright: false` field.
