import io
import json
from pathlib import Path

import pytest

from kratos_smolvla.train import Settings, forward_training_output, training_command


def test_settings_are_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KRATOS_TRAINING_STEPS", "0")
    with pytest.raises(ValueError, match="TRAINING_STEPS"):
        Settings.from_environment()


def test_command_is_offline_friendly_and_disables_wandb() -> None:
    settings = Settings(Path("/kratos/outputs"), steps=2000, batch_size=8, device="cuda")
    command = training_command(settings, Path("/kratos/outputs/train"))
    assert any(
        argument.startswith("--policy.path=/opt/huggingface/hub/models--lerobot--smolvla_base/")
        for argument in command
    )
    assert "--dataset.repo_id=lerobot/svla_so100_pickplace" in command
    assert any(
        argument.startswith(
            "--dataset.root=/opt/huggingface/hub/datasets--lerobot--svla_so100_pickplace/"
        )
        for argument in command
    )
    assert "--dataset.revision=728583b5eaf9e739a7f119e2def466fa1d552402" in command
    assert "--policy.push_to_hub=false" in command
    assert (
        '--rename_map={"observation.images.top":"observation.images.camera1",'
        '"observation.images.wrist":"observation.images.camera2"}' in command
    )
    assert not any(argument.startswith("--policy.dtype=") for argument in command)
    assert "--wandb.enable=false" in command
    assert "--policy.device=cuda" in command
    assert "--num_workers=0" in command
    assert "--persistent_workers=false" in command


def test_lerobot_progress_becomes_bounded_kratos_records(
    capsys: pytest.CaptureFixture[str],
) -> None:
    forward_training_output(io.StringIO("step: 10 loss: 0.25\nstep: 999 loss: 0.10\n"), 100)
    output = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert output == [
        {
            "schema_version": "1.0",
            "record": "progress",
            "step": 10,
            "total_steps": 100,
            "unit": "steps",
        },
        {
            "schema_version": "1.0",
            "record": "metric",
            "name": "train.loss",
            "value": 0.25,
            "step": 10,
        },
        {
            "schema_version": "1.0",
            "record": "progress",
            "step": 100,
            "total_steps": 100,
            "unit": "steps",
        },
        {
            "schema_version": "1.0",
            "record": "metric",
            "name": "train.loss",
            "value": 0.1,
            "step": 100,
        },
    ]
