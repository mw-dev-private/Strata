# In-place Q6 head with production async refill

See also [../B70.md](../B70.md) for every environment variable on this branch and what each one measured.

`STRATA_Q6_PLANES_INPLACE=1` repacks the output head into aligned planes within its existing device allocation and keeps the existing Wide arithmetic/reduction order. It adds no second weight copy. MTP raw subset gathers reconstruct original Q6 rows when necessary. The previous default remains available with the flag unset/0.

The combined binary includes the asynchronous loan refill, which remains enabled separately by `STRATA_PREFILL_REFILL_ASYNC=1`. `STRATA_Q6P_SELFTEST=1` checks real-head 1–8-column bitwise parity and first/middle/last raw rows; use it for verification only, not throughput measurements. Both checks passed in the combined build.

Production since 2026-10-15: stable release built from commit 6d603bc, Q6 planes in-place 1, selftest 0, async-refill 1, refill-verify 0, Tail 7 / Share auto / NativeGraph 1 / GDN 0 / PCIe 0.35, no batch. Actual process and health checked. The previous refill-only release and timestamped config backup remain available.

Short full-server comparison, same starting learned profile and four frozen requests per arm, 512 generated tokens each: weighted decode 58.9251 → 59.7015 tok/s (+1.32 %), both 13,422 cache slots. Prefill practically unchanged. One on-request reached 64.8 tok/s, but this is not the aggregate. A separately predetermined first-fixture request with max 640 naturally ended after 332 tokens at 62.2656 tok/s. These are genuine server outputs, not injected follow tokens. The earlier frozen CLI Q6 comparison at 65 tok/s is a different benchmark; 63+ sustained production throughput has not been established by this short test.

Source integration 6d603bc reuses the verified refill Generate object and Q6 kernel/MTP archives; the combined link recipe is recorded in the build-probe `link-argv` dump. GPU tests must stop/restore production with the existing wrappers; native API requests run through the server. No multi-GPU or long stability validation in this round.
