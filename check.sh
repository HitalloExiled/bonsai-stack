#!/usr/bin/env bash
# bonsai-stack check: serve-bonsai.sh syntax + estimate/dry-run forwarding.
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
tmp="$(mktemp -d)"
trap 'rm -rf -- "$tmp"' EXIT

bash -n "$root/serve-bonsai.sh"

# File math on stubs (needs the fork image for the inspect gate).
if command -v docker >/dev/null 2>&1 && \
   docker image inspect "${NINFER_BONSAI_IMAGE:-bonsai-llama-4090:sm89}" >/dev/null 2>&1; then
  mkdir -- "$tmp/gguf"
  truncate -s 6G "$tmp/gguf/Ternary-Bonsai-2-27B-PTQ1_0.gguf"
  truncate -s 700M "$tmp/gguf/Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf"

  NINFER_BONSAI_DIR="$tmp/gguf" NINFER_NO_VRAM_CHECK=1 \
    "$root/serve-bonsai.sh" --profile quad128 --estimate | grep -qE '~[0-9]+ MiB'
  NINFER_BONSAI_DIR="$tmp/gguf" NINFER_NO_VRAM_CHECK=1 \
    "$root/serve-bonsai.sh" --profile quad128 --dry-run | grep -q -- '--parallel 4'
  NINFER_BONSAI_DIR="$tmp/gguf" NINFER_NO_VRAM_CHECK=1 \
    "$root/serve-bonsai.sh" --profile quad128 --dry-run | grep -q -- '--no-kv-unified'
  NINFER_BONSAI_DIR="$tmp/gguf" NINFER_NO_VRAM_CHECK=1 \
    "$root/serve-bonsai.sh" --profile octo64 --dry-run | grep -q -- '--parallel 8'
  NINFER_BONSAI_DIR="$tmp/gguf" NINFER_NO_VRAM_CHECK=1 \
    "$root/serve-bonsai.sh" --profile max --dry-run | grep -q -- '--parallel 1'
  # Doubled quota: quad128 must carry twice the total context of duo128.
  duo="$(NINFER_BONSAI_DIR="$tmp/gguf" NINFER_NO_VRAM_CHECK=1 \
    "$root/serve-bonsai.sh" --profile duo128 --dry-run)"
  printf '%s' "$duo" | grep -q -- '-c 262144'
  NINFER_BONSAI_DIR="$tmp/gguf" NINFER_NO_VRAM_CHECK=1 \
    "$root/serve-bonsai.sh" --profile quad128 --dry-run | grep -q -- '-c 524288'
  # A stock llama.cpp tag must be refused (exit 4), not launched.
  if docker image inspect ghcr.io/ggml-org/llama.cpp:server-cuda-b11854 >/dev/null 2>&1; then
    rc=0
    NINFER_BONSAI_IMAGE=ghcr.io/ggml-org/llama.cpp:server-cuda-b11854 \
      NINFER_BONSAI_DIR="$tmp/gguf" NINFER_NO_VRAM_CHECK=1 \
      "$root/serve-bonsai.sh" --profile max --dry-run >/dev/null 2>&1 || rc=$?
    [[ "$rc" == "4" ]]
  fi
fi

printf '%s\n' 'bonsai-stack checks passed.'
