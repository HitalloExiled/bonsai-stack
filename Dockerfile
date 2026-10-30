# Bonsai 2 27B ternary runtime: PrismML fork of llama.cpp, CUDA sm_89 (RTX 4090).
#
# Stock llama.cpp cannot run the PTQ1_0/PQ2_0 packs: it rejects PTQ1_0/PQ2_0 as
# unknown types and silently mis-loads legacy Q2_0, emitting garbage. The
# org.prismml.ternary label below is what serve-bonsai.sh gates on.
#
# The fork is the `llama.cpp/` submodule of this repo: its gitlink IS the pin,
# so `git submodule update --init` restores the exact tree and the build needs
# no network. Nothing here hard-codes a commit; the tag is only stamped into
# the image for provenance, and it is read from the gitlink at build time:
#   docker build -f Dockerfile \
#     --build-arg PRISM_TAG="$(git -C llama.cpp describe --tags --exact-match)" \
#     -t bonsai-llama-4090:sm89 .
# (from bonsai-stack/; models/ is excluded by .dockerignore)
ARG CUDA_VERSION=13.1.2
ARG UBUNTU_VERSION=ubuntu24.04

FROM nvidia/cuda:${CUDA_VERSION}-devel-${UBUNTU_VERSION} AS build
# e.g. prism-b10831-9d9efea -> commit 9d9efea in `llama-server --version`.
ARG PRISM_TAG=unknown
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake git ca-certificates \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /src
COPY llama.cpp/ /src/
RUN cmake -S /src -B /build \
        -DCMAKE_BUILD_TYPE=Release \
        -DGGML_CUDA=ON \
        -DGGML_CUDA_NO_VMM=ON \
        -DCMAKE_CUDA_ARCHITECTURES=89 \
        -DLLAMA_CURL=OFF \
        -DLLAMA_BUILD_COMMIT="${PRISM_TAG##*-}" \
        -DLLAMA_BUILD_TESTS=OFF \
        -DLLAMA_BUILD_EXAMPLES=OFF \
        -DLLAMA_BUILD_SERVER=ON \
    && cmake --build /build -j"$(nproc)" --target llama-server

FROM nvidia/cuda:${CUDA_VERSION}-runtime-${UBUNTU_VERSION}
ARG PRISM_TAG=unknown
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 libcurl4 ca-certificates \
    && rm -rf /var/lib/apt/lists/*
LABEL org.prismml.ternary="true" \
      org.prismml.fork="PrismML-Eng/llama.cpp" \
      org.prismml.tag="${PRISM_TAG}"
WORKDIR /app
# Whole dir, not *.so: the libraries are versioned symlinks (libllama.so.0 ->
# libllama.so.0.x.y) and copying only the glob breaks the loader.
COPY --from=build /build/bin/ /app/
ENV LD_LIBRARY_PATH=/app
ENTRYPOINT ["/app/llama-server"]
