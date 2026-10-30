# Bonsai 2 27B ternary runtime: PrismML fork of llama.cpp, CUDA sm_89 (RTX 4090).
#
# Stock llama.cpp cannot run the PTQ1_0/PQ2_0 packs: it rejects PTQ1_0/PQ2_0 as
# unknown types and silently mis-loads legacy Q2_0, emitting garbage. The
# org.prismml.ternary label below is what serve-bonsai.sh gates on.
#
# Build (from the fork checkout next to this file):
#   docker build -f Dockerfile -t bonsai-llama-4090:sm89 .
#
# Pin the fork commit in ./PIN and pass it in so the image is reproducible:
#   docker build -f Dockerfile --build-arg PRISM_COMMIT=$(cat PIN) -t bonsai-llama-4090:sm89 .
ARG CUDA_VERSION=13.1.2
ARG UBUNTU_VERSION=ubuntu24.04

FROM nvidia/cuda:${CUDA_VERSION}-devel-${UBUNTU_VERSION} AS build
ARG PRISM_COMMIT=unknown
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake git ca-certificates \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /src
COPY . /src
RUN cmake -S /src -B /build \
        -DCMAKE_BUILD_TYPE=Release \
        -DGGML_CUDA=ON \
        -DGGML_CUDA_NO_VMM=ON \
        -DCMAKE_CUDA_ARCHITECTURES=89 \
        -DLLAMA_CURL=OFF \
        -DLLAMA_BUILD_TESTS=OFF \
        -DLLAMA_BUILD_EXAMPLES=OFF \
        -DLLAMA_BUILD_SERVER=ON \
    && cmake --build /build -j"$(nproc)" --target llama-server

FROM nvidia/cuda:${CUDA_VERSION}-runtime-${UBUNTU_VERSION}
ARG PRISM_COMMIT=unknown
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 libcurl4 ca-certificates \
    && rm -rf /var/lib/apt/lists/*
LABEL org.prismml.ternary="true" \
      org.prismml.fork="PrismML-Eng/llama.cpp" \
      org.prismml.commit="${PRISM_COMMIT}"
WORKDIR /app
COPY --from=build /build/bin/ /app/
ENV LD_LIBRARY_PATH=/app
ENTRYPOINT ["/app/llama-server"]
