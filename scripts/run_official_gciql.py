"""Run the pinned official OGBench GCIQL implementation."""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path
from typing import Optional

try:
    from setup_official_ogbench import (
        DEFAULT_SOURCE_DIR,
        DEFAULT_VENV_DIR,
        enable_wandb_env_override,
        ensure_source,
        ensure_venv,
    )
except ModuleNotFoundError:
    from scripts.setup_official_ogbench import (
        DEFAULT_SOURCE_DIR,
        DEFAULT_VENV_DIR,
        enable_wandb_env_override,
        ensure_source,
        ensure_venv,
    )


def build_command(
    python: Path,
    *,
    env_name: str,
    seed: int,
    save_dir: Path,
    smoke: bool,
    train_steps: int,
    eval_episodes: int,
    log_interval: int,
    eval_interval: int,
    save_interval: int,
    eval_tasks: Optional[int],
    restore_path: Optional[Path],
    restore_epoch: Optional[int],
) -> list[str]:
    command = [
        str(python),
        "main.py",
        f"--env_name={env_name}",
        f"--seed={seed}",
        f"--save_dir={save_dir}",
        "--run_group=OfficialGCIQL",
        f"--train_steps={train_steps}",
        f"--eval_episodes={eval_episodes}",
        f"--log_interval={log_interval}",
        f"--eval_interval={eval_interval}",
        f"--save_interval={save_interval}",
        "--video_episodes=0",
        "--agent=agents/gciql.py",
        "--agent.actor_p_randomgoal=0.5",
        "--agent.actor_p_trajgoal=0.5",
        "--agent.alpha=0.003",
    ]
    if eval_tasks is not None:
        command.append(f"--eval_tasks={eval_tasks}")
    if restore_path is not None:
        command.append(f"--restore_path={restore_path}")
    if restore_epoch is not None:
        command.append(f"--restore_epoch={restore_epoch}")
    if smoke:
        command.extend(
            [
                "--log_interval=1",
                "--eval_interval=1",
                f"--save_interval={train_steps}",
            ]
        )
    return command


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--venv-dir", type=Path, default=DEFAULT_VENV_DIR)
    parser.add_argument("--env-name", default="pointmaze-medium-stitch-v0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-dir", type=Path, default=Path("runs/official-gciql"))
    parser.add_argument("--train-steps", type=int, default=1000000)
    parser.add_argument("--eval-episodes", type=int, default=50)
    parser.add_argument("--log-interval", type=int, default=5000)
    parser.add_argument("--eval-interval", type=int, default=100000)
    parser.add_argument("--save-interval", type=int, default=100000)
    parser.add_argument("--eval-tasks", type=int, default=None)
    parser.add_argument("--restore-path", type=Path, default=None)
    parser.add_argument("--restore-epoch", type=int, default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--setup", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_dir = ensure_source(args.source_dir) if args.setup else args.source_dir
    python = ensure_venv(args.venv_dir) if args.setup else args.venv_dir / "bin" / "python"
    if not (source_dir / "impls" / "main.py").exists():
        raise FileNotFoundError("Official OGBench source is missing. Run with --setup first.")
    if not python.exists():
        raise FileNotFoundError("Official JAX environment is missing. Run with --setup first.")
    enable_wandb_env_override(source_dir)
    train_steps = 1 if args.smoke else args.train_steps
    eval_episodes = 1 if args.smoke else args.eval_episodes
    command = build_command(
        python.absolute(),
        env_name=args.env_name,
        seed=args.seed,
        save_dir=args.save_dir.resolve(),
        smoke=args.smoke,
        train_steps=train_steps,
        eval_episodes=eval_episodes,
        log_interval=args.log_interval,
        eval_interval=args.eval_interval,
        save_interval=args.save_interval,
        eval_tasks=1 if args.smoke else args.eval_tasks,
        restore_path=None
        if args.restore_path is None
        else args.restore_path.resolve(),
        restore_epoch=args.restore_epoch,
    )
    env = dict(os.environ)
    env["WANDB_MODE"] = "disabled"
    env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    subprocess.run(command, cwd=(source_dir / "impls").absolute(), env=env, check=True)


if __name__ == "__main__":
    main()
