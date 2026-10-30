# bonsai-stack

`llama-server` launcher for **Bonsai 2 27B** ternary GGUF
([prism-ml/Ternary-Bonsai-2-27B-gguf](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf))
on an RTX 4090, via Docker. Companion to `gguf-stack` (stock llama.cpp) and
`ninfer-stack`.

- `serve-bonsai.sh` — profiles max/duo128/quad128/octo64/vision, VRAM pre-flight gate
- `Dockerfile` — PrismML llama.cpp fork build (CUDA sm_89), labelled `org.prismml.ternary`
- `PIN` — fork commit the image was built from
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


## Setup

```bash
# 1. runtime image from the pinned fork
git clone https://github.com/PrismML-Eng/llama.cpp /tmp/bonsai-llama.cpp
git -C /tmp/bonsai-llama.cpp checkout "$(awk 'NR==1{print $3}' PIN)"
docker build -f Dockerfile -t bonsai-llama-4090:sm89 /tmp/bonsai-llama.cpp

# 2. ternary weights (5.95 GB + 0.63 GB mmproj)
hf download prism-ml/Ternary-Bonsai-2-27B-gguf \
    Ternary-Bonsai-2-27B-PTQ1_0.gguf \
    Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf --local-dir models
```

Run `./check.sh` to validate.
