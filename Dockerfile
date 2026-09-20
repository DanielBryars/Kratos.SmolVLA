FROM python:3.12.11-slim-bookworm@sha256:c00fc7b44d844b6da22861ec24af43968a5200eac4ec607b4725d585165d6b49 AS assets

ARG LEROBOT_REVISION=5aa74557f84c54d4b458f8b9643c5aa2982acfed
ARG MODEL_REVISION=d9f33c94a60fb382c90dea2164c96845bd955e28
ARG DATASET_REVISION=728583b5eaf9e739a7f119e2def466fa1d552402
ARG VLM_REVISION=7b375e1b73b11138ff12fe22c8f2822d8fe03467
ARG TORCH_VERSION=2.11.0
ARG TORCHVISION_VERSION=0.26.0

ENV DEBIAN_FRONTEND=noninteractive \
    HF_HOME=/opt/huggingface \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN apt-get update \
    && apt-get install --yes --no-install-recommends ffmpeg git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/huggingface/lerobot.git /opt/lerobot \
    && git -C /opt/lerobot checkout --detach "$LEROBOT_REVISION" \
    && pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cu128 \
        "torch==$TORCH_VERSION" \
        "torchvision==$TORCHVISION_VERSION" \
    && pip install --no-cache-dir "/opt/lerobot[smolvla,training]"

RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('lerobot/smolvla_base', revision='$MODEL_REVISION')" \
    && python -c "from huggingface_hub import snapshot_download; snapshot_download('lerobot/svla_so100_pickplace', repo_type='dataset', revision='$DATASET_REVISION')" \
    && python -c "from huggingface_hub import snapshot_download; snapshot_download('HuggingFaceTB/SmolVLM2-500M-Video-Instruct', revision='$VLM_REVISION')" \
    && mkdir -p /opt/huggingface/hub/models--HuggingFaceTB--SmolVLM2-500M-Video-Instruct/refs \
    && printf '%s' "$VLM_REVISION" > /opt/huggingface/hub/models--HuggingFaceTB--SmolVLM2-500M-Video-Instruct/refs/main

FROM assets AS runtime
WORKDIR /opt/kratos-smolvla
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install --no-cache-dir --no-deps . \
    && useradd --create-home --uid 10001 trainer \
    && mkdir -p /kratos/outputs \
    && chown trainer:trainer /kratos/outputs

ENV HF_HUB_OFFLINE=1 \
    HF_DATASETS_CACHE=/tmp/huggingface/datasets \
    XDG_CACHE_HOME=/tmp/cache \
    TRANSFORMERS_OFFLINE=1 \
    TOKENIZERS_PARALLELISM=false \
    KRATOS_OUTPUT_DIR=/kratos/outputs
USER 10001:10001
ENTRYPOINT ["kratos-smolvla"]
