FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime@sha256:c16f4c749e2d9e96878875cdf6cc45cddda1d1a36fddd371dd6f2360f1b6e2a2 AS assets

ARG LEROBOT_REVISION=5aa74557f84c54d4b458f8b9643c5aa2982acfed
ARG MODEL_REVISION=d9f33c94a60fb382c90dea2164c96845bd955e28
ARG DATASET_REVISION=728583b5eaf9e739a7f119e2def466fa1d552402

ENV DEBIAN_FRONTEND=noninteractive \
    HF_HOME=/opt/huggingface \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN apt-get update \
    && apt-get install --yes --no-install-recommends ffmpeg git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/huggingface/lerobot.git /opt/lerobot \
    && git -C /opt/lerobot checkout --detach "$LEROBOT_REVISION" \
    && pip install --no-cache-dir "/opt/lerobot[smolvla,training]"

RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('lerobot/smolvla_base', revision='$MODEL_REVISION')" \
    && python -c "from huggingface_hub import snapshot_download; snapshot_download('lerobot/svla_so100_pickplace', repo_type='dataset', revision='$DATASET_REVISION')"

FROM assets AS runtime
WORKDIR /opt/kratos-smolvla
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install --no-cache-dir --no-deps . \
    && useradd --create-home --uid 10001 trainer \
    && mkdir -p /kratos/outputs \
    && chown trainer:trainer /kratos/outputs

ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    TOKENIZERS_PARALLELISM=false \
    KRATOS_OUTPUT_DIR=/kratos/outputs
USER 10001:10001
ENTRYPOINT ["kratos-smolvla"]

