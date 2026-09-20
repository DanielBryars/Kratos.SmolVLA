# Kratos SmolVLA

A reproducible, offline Kratos workload that fine-tunes Hugging Face's compact SmolVLA robotics
policy on the SO-100 pick-and-place demonstrations.

## Pinned inputs

| Input | Revision | Licence |
|---|---|---|
| `lerobot/smolvla_base` | `d9f33c94a60fb382c90dea2164c96845bd955e28` | Apache-2.0 |
| `lerobot/svla_so100_pickplace` | `728583b5eaf9e739a7f119e2def466fa1d552402` | Apache-2.0 |
| `huggingface/lerobot` | `5aa74557f84c54d4b458f8b9643c5aa2982acfed` | Apache-2.0 |

The image build downloads those exact revisions into the Hugging Face cache. Runtime sets
`HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`, so the scheduled container needs no network or
Hugging Face token.

## Build

The full image contains the base model and dataset and is intentionally large:

```shell
docker build -t kratos-smolvla:dev .
```

Do not pass a Hugging Face token: every pinned input is public. No credential is written into the
image, build history or repository.

## Test locally on one GPU

Start with a short integration run before spending several hours on the full recipe:

```shell
docker run --rm --gpus all --network none --read-only \
  --security-opt no-new-privileges --cap-drop ALL \
  --tmpfs /tmp:rw,noexec,nosuid,size=2g \
  -e KRATOS_TRAINING_STEPS=1000 \
  -e KRATOS_BATCH_SIZE=8 \
  -v "$PWD/outputs:/kratos/outputs" \
  kratos-smolvla:dev
```

Raise `KRATOS_TRAINING_STEPS` to `20000` only after the short run proves CUDA compatibility,
memory use, output packaging and training-loss telemetry. Tune batch size from evidence; the image
does not assume that one GPU behaves like another.

## Kratos contract

The workload emits newline-delimited Kratos records on stdout:

- pinned model, dataset, step and batch parameters;
- bounded progress and `train.loss` metrics parsed from LeRobot's output; and
- one final structured result.

Human-oriented LeRobot output goes to stderr. Successful runs write:

| Path | Role | Suggested limit |
|---|---|---:|
| `smolvla-checkpoint.tar` | `model` | 8 GiB |
| `run-summary.json` | `metadata` | 64 KiB |

Register both paths as mandatory Kratos outputs when scheduling the job. Kratos computes and
verifies their hashes after the container exits.

## Multi-machine training

LeRobot supports distributed GPU training through Accelerate. Kratos currently assigns one GPU on
one worker to an attempt and starts the container with networking disabled, so a two-machine run is
not enabled by changing this image. It requires Kratos gang scheduling, rank assignment, a scoped
peer network, coordinated leases and all-rank failure handling. The one-GPU run is the acceptance
baseline for that work.

