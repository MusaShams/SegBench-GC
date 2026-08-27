from pathlib import Path

from scripts.run_official_gciql import build_command
from scripts.setup_official_ogbench import ARCHIVE_URL, REQUIREMENTS, REVISION, enable_wandb_env_override


def test_official_revision_and_cpu_requirements_are_pinned() -> None:
    assert REVISION in ARCHIVE_URL
    assert "jax==0.4.30" in REQUIREMENTS
    assert "mujoco==3.3.7" in REQUIREMENTS


def test_official_gciql_command_matches_stitch_protocol() -> None:
    command = build_command(
        Path("/tmp/official/bin/python"),
        env_name="pointmaze-medium-stitch-v0",
        seed=0,
        save_dir=Path("/tmp/results"),
        smoke=False,
        train_steps=1000000,
        eval_episodes=50,
        log_interval=5000,
        eval_interval=100000,
        save_interval=100000,
        eval_tasks=None,
        restore_path=None,
        restore_epoch=None,
    )

    assert "--agent=agents/gciql.py" in command
    assert "--agent.actor_p_randomgoal=0.5" in command
    assert "--agent.actor_p_trajgoal=0.5" in command
    assert "--agent.alpha=0.003" in command
    assert "--train_steps=1000000" in command
    assert "--eval_episodes=50" in command
    assert "--log_interval=5000" in command
    assert "--eval_interval=100000" in command
    assert "--save_interval=100000" in command


def test_wandb_patch_only_enables_environment_override(tmp_path) -> None:
    log_utils = tmp_path / "impls" / "utils" / "log_utils.py"
    log_utils.parent.mkdir(parents=True)
    log_utils.write_text("kwargs = dict(\n        mode=mode,\n)\n", encoding="utf-8")

    enable_wandb_env_override(tmp_path)

    assert "mode=os.environ.get('WANDB_MODE', mode)" in log_utils.read_text(encoding="utf-8")
