# B70 at 131072-token context — the two charts on the front page

Measured 2026-10-17 against the live production server with Strata's own bench
(`python tools/strata_bench.py --long --runs 3`), median of n = 3 per prompt test, thinking off, temperature 0.
Card: Arc Pro B70 32 GB. Model: Qwen3.8-Flash-Next GSQ-RCO IQ3_S, two GGUF shards, `--kv int8`,
13,423 of 24,576 experts cached.

| what | value | range |
| --- | ---: | --- |
| prompt reading, 4K | 1,335 tok/s | 1,321–1,343 |
| prompt reading, 16K | 1,747 tok/s | 1,746–1,752 |
| prompt reading, 32K | 1,744 tok/s | 1,730–1,750 |
| answer speed, short chat | 65.6 tok/s | 62.3–85.6, n = 9 |
| answer speed, agent session | 63.1 tok/s | 48.9–70.0 |
| decode, before the port (0.1.44 line) | 64.17 tok/s | three alternating pairs |
| decode, after the port (this branch) | 63.73 tok/s | three alternating pairs |
| 32K prefill, before | 18,883.0 ms | three alternating pairs |
| 32K prefill, after | 19,005.7 ms | three alternating pairs |

`b70-measured.png` is the first five rows, `b70-ab.png` the last four. Error bars are the measured ranges, not
standard deviations. Both charts are drawn from the table above; the numbers, not the pictures, are the claim.
