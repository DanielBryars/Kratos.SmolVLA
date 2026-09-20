from __future__ import annotations

import json
import os
import re
import subprocess
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

MODEL_ID = "lerobot/smolvla_base"
MODEL_REVISION = "d9f33c94a60fb382c90dea2164c96845bd955e28"
DATASET_ID = "lerobot/svla_so100_pickplace"
DATASET_REVISION = "728583b5eaf9e739a7f119e2def466fa1d552402"
LOSS_PATTERN = re.compile(r"(?:^|\s)loss[:=]\s*(?P<loss>[0-9]+(?:\.[0-9]+)?)", re.IGNORECASE)
STEP_PATTERN = re.compile(r"(?:^|\s)step[:=]\s*(?P<step>[0-9]+)", re.IGNORECASE)


@dataclass(frozen=True)
class Settings:
    output_root: Path
    steps: int
    batch_size: int
    device: str

    @classmethod
    def from_environment(cls) -> Settings:
        settings = cls(
            output_root=Path(os.getenv("KRATOS_OUTPUT_DIR", "/kratos/outputs")),
            steps=int(os.getenv("KRATOS_TRAINING_STEPS", "2000")),
            batch_size=int(os.getenv("KRATOS_BATCH_SIZE", "8")),
            device=os.getenv("KRATOS_DEVICE", "cuda"),
        )
        if not 1 <= settings.steps <= 200_000:
            raise ValueError("KRATOS_TRAINING_STEPS must be between 1 and 200000")
        if not 1 <= settings.batch_size <= 256:
            raise ValueError("KRATOS_BATCH_SIZE must be between 1 and 256")
        if settings.device != "cuda":
            raise ValueError("this workload requires KRATOS_DEVICE=cuda")
        return settings


def emit(record: str, **fields: object) -> None:
    print(json.dumps({"schema_version": "1.0", "record": record, **fields}), flush=True)


def training_command(settings: Settings, training_dir: Path) -> list[str]:
    return [
        "lerobot-train",
        f"--policy.path={MODEL_ID}",
        "--policy.dtype=bfloat16",
        f"--policy.device={settings.device}",
        f"--dataset.repo_id={DATASET_ID}",
        f"--output_dir={training_dir}",
        "--job_name=kratos-smolvla-pickplace",
        f"--steps={settings.steps}",
        f"--batch_size={settings.batch_size}",
        "--wandb.enable=false",
        "--save_freq=500",
        "--log_freq=10",
    ]


def forward_training_output(stream: TextIO, total_steps: int) -> None:
    for line in stream:
        text = line.rstrip("\n")
        print(text, file=os.sys.stderr, flush=True)
        step_match = STEP_PATTERN.search(text)
        loss_match = LOSS_PATTERN.search(text)
        if step_match:
            step = min(int(step_match.group("step")), total_steps)
            emit("progress", step=step, total_steps=total_steps, unit="steps")
            if loss_match:
                emit("metric", name="train.loss", value=float(loss_match.group("loss")), step=step)


def package_checkpoint(training_dir: Path, destination: Path) -> None:
    checkpoints = sorted(training_dir.glob("checkpoints/*"))
    source = checkpoints[-1] if checkpoints else training_dir
    with tarfile.open(destination, "w") as archive:
        archive.add(source, arcname="checkpoint")


def run(settings: Settings) -> int:
    settings.output_root.mkdir(parents=True, exist_ok=True)
    training_dir = settings.output_root / "train"
    emit("param", name="model.id", value=MODEL_ID)
    emit("param", name="model.revision", value=MODEL_REVISION)
    emit("param", name="dataset.id", value=DATASET_ID)
    emit("param", name="dataset.revision", value=DATASET_REVISION)
    emit("param", name="training.steps", value=settings.steps)
    emit("param", name="training.batch_size", value=settings.batch_size)

    process = subprocess.Popen(
        training_command(settings, training_dir),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    forward_training_output(process.stdout, settings.steps)
    exit_code = process.wait()
    if exit_code != 0:
        return exit_code

    checkpoint = settings.output_root / "smolvla-checkpoint.tar"
    package_checkpoint(training_dir, checkpoint)
    summary = {
        "model": {"id": MODEL_ID, "revision": MODEL_REVISION},
        "dataset": {"id": DATASET_ID, "revision": DATASET_REVISION},
        "steps": settings.steps,
        "batch_size": settings.batch_size,
        "checkpoint": checkpoint.name,
    }
    (settings.output_root / "run-summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    emit("result", result=summary)
    return 0


def main() -> None:
    raise SystemExit(run(Settings.from_environment()))


if __name__ == "__main__":
    main()
