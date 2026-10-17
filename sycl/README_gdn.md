# Intel's GDN delta-rule stage on the Arc Pro B70

The code comments in `sycl/src/program/generate.cpp`, `sycl/src/prefill/kernels.dp.cpp` and
`sycl/src/prefill/intel_gdn/` point here. This is the Intel delta-rule ("GDN") prompt path: Intel's own kernel,
built for this card, as an alternative to Strata's delta-rule stage.

**Status on this branch: built and parity-verified, but off in production.** No engine-level speed win was measured
for it, and it costs device memory (below). Everything is gated at run time, so with the variables unset Strata uses
its own stage and behaves as upstream.

## Build

| CMake option | default | effect |
| --- | --- | --- |
| `STRATA_INTEL_GDN` | `ON` | compiles Intel's GDN delta kernel. If the cutlass-sycl tree is not found, CMake prints `Intel GDN delta kernel: OFF (no cutlass-sycl at …; STRATA_GDN_INTEL stays inert)` and the run-time variable does nothing. |
| `STRATA_CUTLASS_SYCL_DIR` | from `-D` or `$STRATA_CUTLASS_SYCL_DIR` | path to the cutlass-sycl source tree (`include/`, `tools/util/include/`, `applications/`); the check is for `include/cute/tensor.hpp`. |

The kernel is device code for the Battlemage target (`bmg-g31`), built with the same AOT settings as the rest of
`sycl/`.

## Run time

| variable | default | meaning |
| --- | --- | --- |
| `STRATA_GDN_INTEL` | off | `=1` selects Intel's stage for prompt processing. Only then does the engine reserve the stage's scratch (see below). |
| `STRATA_GDN_INTEL_OUTWR` | off | `=1` is the A/B fallback to the old separate output write. The default lets the out-norm kernel read the stage's **FP16** output directly: the FP32 copy carried no extra bits (the delta kernel writes FP16), the norm maths and the rounding are identical, so `y16` is bit-identical to `kernel → out_write → norm`. Guarded by md5 equality in the A/B window. |
| `STRATA_GDN_WS_MEMSET` | off | `=1` restores zeroing the A/w/u workspaces before the kernel starts. Kept as the fallback under that window; the audit below says it is not needed. |
| `STRATA_GDN_WS_POISON` | off | `=1` fills A/w/u with FP16 quiet NaN (`0x7E00`) instead of zeros. This is the proof, not a fallback. |

### The scratch reservation is not optional

`STRATA_GDN_INTEL=1` makes the stage allocate its scratch — padded q/k/v, output, A/w/u — on the **first chunk**,
about **63.9 KB per padded token**: ~1.05 GB for a 16,384-token chunk, and ~2,064 MiB for the ceiling of
`--prefill auto:32768`. The engine adds that to the prefill reservation up front
(`strata generate: STRATA_GDN_INTEL: reserving … MiB`). Without it the expert cache takes the space first and the
first chunk fails with `UR_RESULT_ERROR_OUT_OF_RESOURCES`. A one-shot `generate` caps the reservation to its own
prompt length; `serve` keeps the ceiling.

## Why the workspace zeroing was removed, and how that was proven

The upstream shape of this stage zeroes A, w and u before the kernel runs (Intel's reference uses `torch::zeros`).
The kernel audit shows every cell is written before it is read:

- **A** — `compute_A_kernel` writes each whole 64×64 cell before the inverse is taken (clear, then `K·Kᵀ`, then the
  diagonal and the upper triangle set explicitly, then the whole tile written back unconditionally).
- **u** — `chunk_compute_wu` writes the whole 64×128 tile per head and chunk before the forward pass reads it.
- **w** — the same, behind a guard `chunk != 0 || has_initial_state`. The engine always passes an initial state, so
  the whole tile is written; with `has_initial_state = false`, chunk 0 would need it zeroed.
- Rows `[T, TP+63)` of the padded window are never indexed by any kernel (chunk count is `ceil(seq_len/64)`, stride
  `total_virtual_seqlen = TP`); Intel's zeroing there is an allocation artifact.

So zeroing is mandatory **only** when no initial state is passed. `STRATA_GDN_WS_MEMSET=1` is the fallback if that
conclusion is ever wrong.

Zero-filling alone does not prove it, because freshly allocated memory can contain accidental zeros and the
bench's finiteness check cannot tell "the gate values blew up" from "the kernel read unwritten cells".
`STRATA_GDN_WS_POISON=1` (or `intel_gdn_parity --poison`) fills A/w/u with FP16 NaN instead: if the kernel ever read
an unwritten cell, the NaN would propagate into `iout`, `y` or the state and parity would fail. **It passes**, so the
zero fill is provably unnecessary.

## Checking it

```
intel_gdn_parity --selftest                     # host-side self test
intel_gdn_parity                                # parity against $STRATA_GDN_REF (default ref/gdn_ref_T256.bin)
intel_gdn_parity --ref <file> --expect <file>   # write the staged buffers as the expected file (no GPU needed)
intel_gdn_parity --poison                       # the NaN audit above
gdn_stage_bench                                 # stage-by-stage byte-exact timings (bench/gdn_stage_bench.cpp)
```

`--no-stagecheck` skips the staged-buffer comparison when only the kernel is of interest.
