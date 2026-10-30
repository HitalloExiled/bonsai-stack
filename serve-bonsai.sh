#!/usr/bin/env bash
# serve-bonsai.sh: llama-server launcher for the TERNARY Bonsai 2 27B GGUF
# (PrismML fork of llama.cpp, RTX 4090). Companion to gguf-stack/serve-llama.sh
# (stock llama.cpp) and ninfer-stack/serve.sh.
#
# The ternary packs (PTQ1_0/PQ2_0) are Prism-private ggml types: stock
# llama.cpp refuses them and, worse, loads a legacy Q2_0 *without warning*
# and produces garbage. The image gate below is therefore not a convenience.
#
# Usage:
#   serve-bonsai.sh [--profile max|duo256|trio256|duo128|quad128|penta128|hexa128|octo64|vision]
#                   [--port N] [--host H] [--context N] [--parallel N]
#                   [--estimate] [--dry-run] [--no-vram-check] [--list] [--help]
#                   [-- <extra llama-server args...>]
#
# Profiles (ternary weights ~5.95GB leave room for double the GGUF slots):
#   max:     1 slot  262K ctx, q4_0 KV
#   duo256:  2 slots 256K ctx, q4_0 KV            (two full-window sessions)
#   trio256: 3 slots 256K ctx, q4_0 KV            (the 24GB ceiling: 4x256K
#                                                  needs ~25.4 GiB, plus the
#                                                  ~1.5 GiB the desktop holds)
#   duo128:  2 slots 128K ctx, q4_0 KV            (= gguf-duo128 parity)
#   quad128: 4 slots 128K ctx, q4_0 KV            (= 2x gguf-duo128 quota)
#   penta128:5 slots 128K ctx, q4_0 KV            (~19.5 GiB with mmproj)
#   hexa128: 6 slots 128K ctx, q4_0 KV            (128K ceiling: 7 would need
#                                                  ~24.2 GiB + desktop)
#   octo64:  8 slots  64K ctx, q4_0 KV            (= 2x gguf-quad quota)
#   vision:  1 slot   64K ctx, q8_0 KV + mmproj
#
# Exit codes: 0 ok, 1 usage, 2 missing image/model, 3 insufficient VRAM,
#             4 wrong runtime (stock llama.cpp image).
set -euo pipefail
export LC_ALL=C LC_NUMERIC=C

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
gguf_dir_default="$root/models"
# Fork build from the llama.cpp/ submodule (tag prism-b10831-9d9efea). Never
# point this at a stock ghcr.io/ggml-org image: it loads the legacy Q2_0
# silently and returns garbage.
docker_image="${NINFER_BONSAI_IMAGE:-bonsai-llama-4090:sm89}"
base_file="Ternary-Bonsai-2-27B-PTQ1_0.gguf"
mmproj_file_name="Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf"
mmproj_file=""

usage() {
  cat <<USAGE
usage: $(basename -- "$0") [--profile max|duo256|trio256|duo128|quad128|octo64|vision] [--port N]
       [--host H] [--context N] [--parallel N]
       [--estimate] [--dry-run] [--no-vram-check] [--list] [--help]
       [-- <extra llama-server args...>]

  --profile max|duo256|trio256|duo128|quad128|penta128|hexa128|octo64|vision  preset (default: quad128)
                            max:     1 slot,  262144 ctx, q4_0 KV
                            duo256:  2 slots, 262144 ctx, q4_0 KV
                            trio256: 3 slots, 262144 ctx, q4_0 KV
                            duo128:  2 slots, 131072 ctx, q4_0 KV
                            quad128: 4 slots, 131072 ctx, q4_0 KV
                            penta128:5 slots, 131072 ctx, q4_0 KV
                            hexa128: 6 slots, 131072 ctx, q4_0 KV
                            octo64:  8 slots,  65536 ctx, q4_0 KV
                            vision:  1 slot,   65536 ctx, q8_0 KV + mmproj
  --port N        listen port (default: 18091)
  --host H        bind host for the log line (container binds 0.0.0.0)
  --context N     override TOTAL context tokens (pool split across slots)
  --parallel N    parallel slots (default: per profile, 1 for max/vision)
  --mmproj on|off|auto  vision encoder: on (vision profile always on),
                        off, or auto=off (router upgrades on first image)
  --estimate      print VRAM estimate for the resolved setup and exit 0
  --dry-run       print the docker command and exit 0
  --no-vram-check skip the VRAM pre-flight gate
  --list          list available .gguf files and exit 0
  -- <args...>    extra args appended to llama-server

Ternary g128 weights: {-1,0,+1} per 128 weights, one FP16 scale. Deployed
size 5.95GB at 1.75 bits/weight, 98.2% of the FP16 benchmark average.
KV per-token is q4_0 ~28MiB/1k (same hybrid-attention backbone as Qwen3.8).
USAGE
}

profile="quad128"
port="${NINFER_BONSAI_PORT:-18091}"
host="127.0.0.1"
ctx_override=""
parallel_slots=""
mmproj_mode="auto"
estimate_only=0
dry_run=0
vram_check=1
if [[ "${NINFER_NO_VRAM_CHECK:-0}" == "1" ]]; then vram_check=0; fi
passthrough=()

while (( $# )); do
  case "$1" in
    --help|-h) usage; exit 0 ;;
    --profile) profile="${2:?--profile needs max|duo128|quad128|octo64|vision}"; shift 2 ;;
    --profile=*) profile="${1#--profile=}"; shift ;;
    --port) port="${2:?--port needs a value}"; shift 2 ;;
    --port=*) port="${1#--port=}"; shift ;;
    --host) host="${2:?--host needs a value}"; shift 2 ;;
    --host=*) host="${1#--host=}"; shift ;;
    --context) ctx_override="${2:?--context needs tokens}"; shift 2 ;;
    --context=*) ctx_override="${1#--context=}"; shift ;;
    --parallel) parallel_slots="${2:?--parallel needs slots}"; shift 2 ;;
    --parallel=*) parallel_slots="${1#--parallel=}"; shift ;;
    --mmproj) mmproj_mode="${2:?--mmproj needs on|off|auto}"; shift 2 ;;
    --mmproj=*) mmproj_mode="${1#--mmproj=}"; shift ;;
    --estimate) estimate_only=1; shift ;;
    --dry-run) dry_run=1; shift ;;
    --no-vram-check) vram_check=0; shift ;;
    --list)
      shopt -s nullglob
      for f in "${NINFER_BONSAI_DIR:-$gguf_dir_default}"/*.gguf; do
        printf '%s\n' "$(basename -- "$f")"
      done
      exit 0 ;;
    --) shift; while (( $# )); do passthrough+=("$1"); shift; done ;;
    --*) printf 'Unknown option: %s\n' "$1" >&2; usage >&2; exit 1 ;;
    *) printf 'Unexpected arg (no positional MODEL; files are per-profile): %s\n' "$1" >&2; exit 1 ;;
  esac
done

gguf_dir="${NINFER_BONSAI_DIR:-$gguf_dir_default}"
# Multimodal is lazy: the mmproj encoder (+0.63GB resident) loads only for the
# vision profile or an explicit --mmproj on. The router upgrades a running
# text tier in place on the first image request instead.
if [[ "$profile" == "vision" || "$mmproj_mode" == "on" ]]; then
  mmproj_file="$gguf_dir/$mmproj_file_name"
elif [[ "$mmproj_mode" != "off" && "$mmproj_mode" != "auto" ]]; then
  printf 'Unknown --mmproj: %s (want on|off|auto)\n' "$mmproj_mode" >&2
  exit 1
fi

# ctx is the TOTAL pool (slots share it split evenly); per-slot = ctx/slots.
case "$profile" in
  max)
    ctx=262144; slots=1; kv_type="q4_0"
    profile_label="1 slot, 262144 total ctx, q4_0 KV" ;;
  duo256)
    ctx=524288; slots=2; kv_type="q4_0"
    profile_label="2 slots, 262144 ctx each, q4_0 KV" ;;
  trio256)
    ctx=786432; slots=3; kv_type="q4_0"
    profile_label="3 slots, 262144 ctx each, q4_0 KV" ;;
  duo128)
    ctx=262144; slots=2; kv_type="q4_0"
    profile_label="2 slots, 131072 ctx each, q4_0 KV" ;;
  quad128)
    ctx=524288; slots=4; kv_type="q4_0"
    profile_label="4 slots, 131072 ctx each, q4_0 KV" ;;
  penta128)
    ctx=655360; slots=5; kv_type="q4_0"
    profile_label="5 slots, 131072 ctx each, q4_0 KV" ;;
  hexa128)
    ctx=786432; slots=6; kv_type="q4_0"
    profile_label="6 slots, 131072 ctx each, q4_0 KV" ;;
  octo64)
    ctx=524288; slots=8; kv_type="q4_0"
    profile_label="8 slots, 65536 ctx each, q4_0 KV" ;;
  vision)
    ctx=65536; slots=1; kv_type="q8_0"
    profile_label="1 slot + mmproj, 65536 ctx, q8_0 KV" ;;
  *) printf 'Unknown --profile: %s (want max|duo256|trio256|duo128|quad128|penta128|hexa128|octo64|vision)\n' "$profile" >&2; exit 1 ;;
esac
if [[ -n "$parallel_slots" ]]; then slots="$parallel_slots"; fi
if [[ -n "$ctx_override" ]]; then ctx="$ctx_override"; fi

if ! command -v docker >/dev/null 2>&1; then
  printf 'error: docker not found\n' >&2
  exit 2
fi
if ! docker image inspect "$docker_image" >/dev/null 2>&1; then
  printf 'Missing image %s; build it with: %s/Dockerfile\n' "$docker_image" "$root" >&2
  exit 2
fi
# Refuse a stock llama.cpp image: it would load Q2_0 silently and emit garbage.
if [[ -z "$(docker image inspect -f '{{ index .Config.Labels "org.prismml.ternary" }}' "$docker_image" 2>/dev/null)" ]]; then
  printf 'error: %s is not a PrismML ternary build (missing label org.prismml.ternary)\n' "$docker_image" >&2
  printf 'Stock llama.cpp cannot run PTQ1_0/PQ2_0. Use the fork image.\n' >&2
  exit 4
fi
if [[ ! -f "$gguf_dir/$base_file" ]]; then
  printf 'Missing model: %s\n' "$gguf_dir/$base_file" >&2
  printf 'Download the PTQ1_0 GGUF into %s first.\n' "$gguf_dir" >&2
  exit 2
fi
if [[ -n "$mmproj_file" && ! -f "$mmproj_file" ]]; then
  printf 'Missing mmproj encoder: %s\n' "$mmproj_file" >&2
  exit 2
fi

# Per-token KV estimate (MiB per 1024 tokens) by llama cache type.
# Calibrated 2026-10-30 against this GGUF on the RTX 4090: the qwen35
# backbone is hybrid, KV only exists on 16 of 64 layers
# (full_attention_interval=4), 4 KV heads x 256 dims -> 18 MiB/1k at q4_0.
# Measured: 1x256K + mmproj = 12,222 MiB; 4x128K = 16.5 GiB; 8x64K = 17.0 GiB.
# The old 28/96 numbers overestimated by ~55% and would have wrongly rejected
# the 3x256K profile in the VRAM gate.
kv_mib_per_1k() {
  case "$1" in
    q8_0) printf '34' ;;
    q4_0) printf '18' ;;
    *) printf '34' ;;
  esac
}

# required_mib <file> <total_ctx> <kv_type> [extra_weights] [slots] -> "total|weights|kv".
# ctx is the TOTAL pool: slots split it, so KV cost is per token, not per slot.
# Each extra slot adds ~128 MiB of recurrent (SSM) state, measured from the
# quad128/octo64 delta (512 MiB across four extra slots).
required_mib() {
  local f="$1" cap="$2" dtype="$3" extra="${4:-}" nslots="${5:-1}"
  local bytes per_1k kv_mib weights_mib overhead total
  bytes="$(stat -c%s -- "$f" 2>/dev/null || wc -c < "$f" | tr -d ' ')"
  if [[ -n "$extra" ]]; then
    bytes=$((bytes + $(stat -c%s -- "$extra" 2>/dev/null || wc -c < "$extra" | tr -d ' ')))
  fi
  per_1k="$(kv_mib_per_1k "$dtype")"
  kv_mib="$(awk -v c="$cap" -v k="$per_1k" \
    'BEGIN { printf "%d", (c / 1024) * k }')"
  weights_mib="$(awk -v b="$bytes" 'BEGIN { printf "%d", (b / 1048576) * 0.99 }')"
  # CUDA context/workspace/graphs (~1.3 GiB) plus per-slot SSM state.
  overhead=$((1536 + nslots * 128))
  total=$((weights_mib + kv_mib + overhead))
  printf '%d|%d|%d' "$total" "$weights_mib" "$kv_mib"
}

estimate="$(required_mib "$gguf_dir/$base_file" "$ctx" "$kv_type" "$mmproj_file" "$slots")"
need_mib="${estimate%%|*}"; rest="${estimate#*|}"
weights_mib="${rest%%|*}"; kv_mib="${rest##*|}"

if [[ "$estimate_only" == "1" ]]; then
  overhead_mib=$((need_mib - weights_mib - kv_mib))
  printf 'Estimate: weights ~%d MiB + kv ~%d MiB (%s %s tokens) + overhead %d MiB = ~%d MiB (%.1f GiB)\n' \
    "$weights_mib" "$kv_mib" "$ctx" "$kv_type" "$overhead_mib" "$need_mib" \
    "$(awk -v m="$need_mib" 'BEGIN { printf "%.1f", m / 1024 }')"
  exit 0
fi

# VRAM pre-flight gate (exit 3 when short).
if [[ "$vram_check" == "1" ]]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    read -r free_mib total_mib <<< "$(nvidia-smi --query-gpu=memory.free,memory.total \
      --format=csv,noheader,nounits | head -1 | tr -d ' ' | tr ',' ' ' || true)"
    if [[ -n "${free_mib:-}" && -n "${total_mib:-}" ]]; then
      printf 'VRAM: free %d MiB / total %d MiB; need ~%d MiB (weights ~%d + kv ~%d + slots/overhead %d)\n' \
        "$free_mib" "$total_mib" "$need_mib" "$weights_mib" "$kv_mib" \
        "$((need_mib - weights_mib - kv_mib))"
      if (( need_mib > total_mib )); then
        printf 'error: setup needs more than total VRAM\n' >&2
        exit 3
      fi
      if (( need_mib > free_mib )); then
        printf 'error: insufficient free VRAM: %d MiB free, ~%d MiB required\n' \
          "$free_mib" "$need_mib" >&2
        printf 'Free VRAM (another backend resident?) or set NINFER_NO_VRAM_CHECK=1.\n' >&2
        exit 3
      fi
      if (( need_mib + 1024 > free_mib )); then
        printf 'warning: <1 GiB headroom after launch\n' >&2
      fi
    else
      printf 'warning: nvidia-smi unreadable; skipping VRAM check\n' >&2
    fi
  else
    printf 'warning: nvidia-smi not found; skipping VRAM check\n' >&2
  fi
fi

model_basename="$(basename -- "$base_file")"
args=(
  -m "/models/$model_basename"
  --alias "bonsai-$profile"
  -ngl 99 -c "$ctx" --parallel "$slots"
  --cache-type-k "$kv_type" --cache-type-v "$kv_type"
  -fa on --jinja
  --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.05
  --host 0.0.0.0 --port "$port"
)
if [[ -n "$mmproj_file" ]]; then
  args+=(--mmproj "/models/$(basename -- "$mmproj_file")")
fi
if (( slots > 1 )); then
  # Independent per-slot KV arenas: a shared unified cache lets one long
  # prompt stall the other slots (the "queueing instead of concurrency" shape).
  args+=(--no-kv-unified)
fi
if (( ${#passthrough[@]} )); then
  args+=("${passthrough[@]}")
fi

if [[ "$dry_run" == "1" ]]; then
  printf 'docker run --gpus all --rm -p %s:%s -v %s:/models:ro %s' \
    "$port" "$port" "$gguf_dir" "$docker_image"
  printf ' %q' "${args[@]}"
  printf '\n'
  exit 0
fi

printf 'Starting llama-server Bonsai 2 27B ternary (%s) at http://%s:%s/v1\n' \
  "$profile_label" "$host" "$port"
exec docker run --gpus all --rm --init -p "$port:$port" \
  -v "$gguf_dir:/models:ro" "$docker_image" "${args[@]}"
