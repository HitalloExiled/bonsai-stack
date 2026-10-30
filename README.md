# bonsai-stack

`llama-server` launcher for **Bonsai 2 27B** ternary GGUF
([prism-ml/Ternary-Bonsai-2-27B-gguf](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf))
on an RTX 4090, via Docker. Companion to `gguf-stack` (stock llama.cpp) and
`ninfer-stack`.

- `serve-bonsai.sh` — profiles max/duo128/quad128/octo64/vision, VRAM pre-flight gate
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
| `max` | 1 | 262144 | 256K | q4_0 | ~14.0 GiB | — |
| `duo128` | 2 | 131072 | 256K | q4_0 | ~14.0 GiB | — |
| `quad128` | 4 | 131072 | 512K | q4_0 | ~21.0 GiB | 16.5 GiB |
| `octo64` | 8 | 65536 | 512K | q4_0 | ~21.0 GiB | 17.0 GiB |
| `vision` | 1 | 65536 | 64K + mmproj | q8_0 | ~9.4 GiB | — |

`quad128` is 2x the `gguf-duo128` quota; `octo64` is 2x the `gguf-quad` quota.
Measured on the RTX 4090 (`nvidia-smi` after boot): the launcher estimate is a
conservative upper bound. Concurrency verified with
`scripts/parallel-probe.py`: 4/4 and 8/8 slots busy simultaneously, ~68-83
tok/s per stream at 4-way.

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
