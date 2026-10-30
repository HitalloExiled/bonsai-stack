# bonsai-stack

`llama-server` launcher for **Bonsai 2 27B** ternary GGUF
([prism-ml/Ternary-Bonsai-2-27B-gguf](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf))
on an RTX 4090, via Docker. Companion to `gguf-stack` (stock llama.cpp) and
`ninfer-stack`.

- `serve-bonsai.sh` — profiles max/duo256/trio256/duo128/quad128/penta128/hexa128/octo64/vision, VRAM pre-flight gate
- `Dockerfile` — PrismML llama.cpp fork build (CUDA sm_89), labelled `org.prismml.ternary`
- `llama.cpp/` — PrismML fork submodule, tagged `prism-b10831-9d9efea` (the gitlink is the pin)
- `models/` — local `.gguf` artifacts (git-ignored)

## Why a separate stack

The ternary packs (`PTQ1_0` at 1.75 bpw, `PQ2_0` at 2.13 bpw) are
Prism-private ggml types. Stock llama.cpp rejects them — and silently
mis-loads legacy `Q2_0`, emitting garbage. The launcher refuses any image
without the `org.prismml.ternary` label (exit 4).

## Profiles

Ternary weights are ~5.95 GB instead of the Qwen3.8 IQ3_S ~11 GB, so the
slot quota doubles at the same 24 GB card budget:

| profile | slots | ctx/slot | total ctx | KV | est. VRAM | measured |
|---|---|---|---|---|---|---|
| `max` | 1 | 262144 | 256K | q4_0 | ~11.6 GiB | 12.2 GiB (+mmproj) |
| `duo256` | 2 | 262144 | 512K | q4_0 | ~16.2 GiB | — |
| `trio256` | 3 | 262144 | 768K | q4_0 | ~20.9 GiB | 21.0 GiB (+mmproj) |
| `duo128` | 2 | 131072 | 256K | q4_0 | ~11.7 GiB | — |
| `quad128` | 4 | 131072 | 512K | q4_0 | ~16.5 GiB | 16.5 GiB |
| `penta128` | 5 | 131072 | 640K | q4_0 | ~19.4 GiB | 18.8 GiB (+mmproj) |
| `hexa128` | 6 | 131072 | 768K | q4_0 | ~21.2 GiB | 21.2 GiB (+mmproj) |
| `octo64` | 8 | 65536 | 512K | q4_0 | ~17.0 GiB | 17.0 GiB |
| `vision` | 1 | 65536 | 64K + mmproj | q8_0 | ~9.8 GiB | — |

`quad128` is 2x the `gguf-duo128` quota; `octo64` is 2x the `gguf-quad` quota.
Measured on the RTX 4090 (`nvidia-smi` after boot): the launcher estimate is
within ~1% of the real footprint. Concurrency verified with
`scripts/parallel-probe.py`: 4/4 and 8/8 slots busy simultaneously, ~68-83
tok/s per stream at 4-way.

### Maximum parallelism on the 24 GB card

The `qwen35` backbone is hybrid: only 16 of 64 layers carry attention KV
(`full_attention_interval=4`), so a q4_0 token costs **18 MiB/1k** rather than
the 28 MiB/1k of a dense 27B. That is why the ternary tiers can hold more than
the GGUF stack at the same quality.

Measured container footprint (booted alone, `nvidia-smi` after healthy):

| profile | slots × ctx | total ctx | container | GPU free |
|---|---|---|---|---|
| `max` + mmproj | 1 × 256K | 256K | 12.2 GiB | ~10.8 GiB |
| `penta128` + mmproj | 5 × 128K | 640K | 18.8 GiB | ~3.6 GiB |
| `trio256` + mmproj | 3 × 256K | 768K | 21.0 GiB | ~1.3 GiB |
| `hexa128` + mmproj | 6 × 128K | 768K | 21.2 GiB | ~1.1 GiB |

The frontier at full windows:

| slots × ctx | KV | total | fits 24.5 GiB |
|---|---|---|---|
| 2 × 256K | 9.0 GiB | ~16.3 GiB | yes, ~6 GiB spare |
| **3 × 256K** | **13.5 GiB** | **~21.0 GiB** | **yes, ~1.3 GiB spare** |
| 4 × 256K | 18.0 GiB | ~25.6 GiB | no |
| 5 × 128K | 11.3 GiB | 18.8 GiB | yes, ~3.6 GiB spare |
| **6 × 128K** | **13.5 GiB** | **21.2 GiB** | **yes, ~1.1 GiB spare** |
| 7 × 128K | 15.8 GiB | ~23.6 GiB | no (with the desktop) |

`trio256` and `hexa128` are the two ceilings, for different window needs:
three concurrent 256K sessions, or six concurrent 128K sessions, both at
~21 GiB. A fourth 256K slot needs ~25.6 GiB before the ~1.5-1.8 GiB the
desktop already holds; a seventh 128K slot needs ~23.6 GiB + desktop.

`hexa128` was stress-tested with six simultaneous 8K-token prefills: peak
23,601 MiB (963 MiB free), no errors — it fits, but the margin is thin. Keep
`penta128` (3.6 GiB free) if the desktop may need VRAM. Keeping the full
window matters for agent sessions around 200K tokens: 4 × 192K would cost the
same memory as 3 × 256K but reject those prompts, and a 200K request always
forces a switch away from any 128K tier (the router checks this).

Prefill is not a Bonsai penalty: 3.2k tok/s at an 11.4k-token prompt on
`quad128`, against 3.1k tok/s for the Qwen3.8 27B IQ3_S GGUF on the same card.
Small prompts look slower only because of the fixed ~100 ms first-batch cost
(~850 tok/s at 91 tokens), which the GGUF stack pays identically. Slot count
does not change prefill (`max` 3.2k vs `quad128` 3.2k).


## Setup

```bash
# 1. runtime image, built from the pinned fork submodule (no network, no /tmp)
git submodule update --init --recursive          # inside the model-router clone
docker build -f Dockerfile \
    --build-arg PRISM_TAG="$(git -C llama.cpp describe --tags --exact-match)" \
    -t bonsai-llama-4090:sm89 .

# 2. ternary weights (5.95 GB + 0.63 GB mmproj)
hf download prism-ml/Ternary-Bonsai-2-27B-gguf \
    Ternary-Bonsai-2-27B-PTQ1_0.gguf \
    Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf --local-dir models
```

## Runtime pin

The submodule gitlink points at the PrismML release tag
**`prism-b10831-9d9efea`** (2026-10-29, `9d9efea`), the newest `prism-b*`
build. `git -C llama.cpp describe --tags --exact-match` prints it, and `--version`
reports `commit 9d9efea`, so the running container can be traced back to the tag.

The `stable` / `v0.0.1-prism` (2026-03-02) and `v0.0.2-prism` (2026-03-18) tags
are older than the PTQ1_0 pack: `v0.0.2-prism` does not even contain
`GGML_TYPE_PTQ1_0`, so it cannot load this model. The `prism-b*` tags are the
actual release line.

`llama.cpp/` is a submodule of this stack; **its gitlink is the pin**. Restore
it with `git submodule update --init --recursive` and the image rebuilds from
the exact same tree, offline. To move the pin, check out another `prism-b*`
tag inside `llama.cpp`, rebuild, and commit the new gitlink — nothing else in
this repo stores a commit.

Run `./check.sh` to validate.
