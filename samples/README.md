# Samples

Audio8-TTS-Realistic 0.6B cloning from her own reference folders — the same
`reference.wav` and `reference.txt` pairs the project already keeps, one per
language. Nothing was cherry-picked: each file is the first take at seed 123.

| file | language | reference | note |
|---|---|---|---|
| `en.wav` | English | `samantha2` | rendered on CPU |
| `fr.wav` | French | `samantha2-fr` | rendered on CPU |
| `es.wav` | Spanish | `samantha2-es` | rendered on CPU |
| `fr-long.wav` | French, three sentences | `samantha2-fr` | 12.9 s, **1.02x realtime** |
| `fr-court.wav` | French, two short ones | `samantha2-fr` | 2.2 s, **1.18x realtime** |

The grafts do not change the waveform, so a take rendered before them and one
rendered after are the same file. These are here to judge the **voice**, not
the speedup — for the speedup, see `bench/verify.py` in the repository root.

What to listen for: whether it is *her*, across three languages, from three
different reference recordings. Timbre similarity on ECAPA against the
matching reference is 0.740 / 0.723 / 0.689, where two of her own references
compared to each other score only 0.41 to 0.52.
