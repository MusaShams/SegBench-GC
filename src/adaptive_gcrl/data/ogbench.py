"""OGBench adapter with lazy optional dependency handling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from adaptive_gcrl.data.offline_dataset import goal_relabel_batch_from_dataset, transition_batch_from_dataset
from adaptive_gcrl.data.replay_buffer import TransitionBatch


@dataclass(frozen=True)
class OGBenchSpec:
    task: str
    dataset: str = "default"
    observation_mode: str = "state"
    goal_reward_mode: str = "dense_negative_distance"
    success_threshold: float = 1.0

    @property
    def dataset_name(self) -> str:
        return self.task if self.dataset in {"", "default"} else self.dataset


def load_ogbench_dataset(spec: OGBenchSpec) -> Any:
    try:
        import ogbench  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ImportError(
            "OGBench is not installed. Install benchmark extras or follow the OGBench setup instructions "
            "before running real benchmark experiments. On macOS/Python 3.9, use the pinned MuJoCo wheel in the "
            "project benchmark extras to avoid building MuJoCo from source."
        ) from exc

    if not hasattr(ogbench, "make_env_and_datasets"):
        raise AttributeError("Installed ogbench package does not expose make_env_and_datasets.")
    return ogbench.make_env_and_datasets(spec.dataset_name, compact_dataset=False)


def load_ogbench_env(spec: OGBenchSpec) -> Any:
    try:
        import ogbench  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ImportError(
            "OGBench is not installed. Install the pinned benchmark extra before rollout evaluation."
        ) from exc
    if not hasattr(ogbench, "make_env_and_datasets"):
        raise AttributeError("Installed ogbench package does not expose make_env_and_datasets.")
    return ogbench.make_env_and_datasets(spec.dataset_name, env_only=True)


def load_ogbench_transition_batch(spec: OGBenchSpec, seed: int = 0) -> TransitionBatch:
    """Load the train split from OGBench and normalize it into `TransitionBatch`."""

    loaded = load_ogbench_dataset(spec)
    if not isinstance(loaded, tuple) or len(loaded) < 2:
        raise TypeError("Expected ogbench.make_env_and_datasets to return at least (env, train_dataset).")
    train_dataset = loaded[1]
    if not isinstance(train_dataset, dict):
        raise TypeError("Expected OGBench train dataset to be a dictionary-like mapping.")
    if "rewards" not in train_dataset or ("goals" not in train_dataset and "desired_goals" not in train_dataset):
        return goal_relabel_batch_from_dataset(
            train_dataset,
            seed=seed,
            reward_mode=spec.goal_reward_mode,
            success_threshold=spec.success_threshold,
        )
    return transition_batch_from_dataset(train_dataset)
