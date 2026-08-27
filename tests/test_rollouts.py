import numpy as np

from adaptive_gcrl.evaluation.rollouts import evaluate_goal_conditioned_policy


class FakeActionSpace:
    low = np.array([-1.0])
    high = np.array([1.0])


class FakeGoalEnv:
    action_space = FakeActionSpace()

    def __init__(self) -> None:
        self.steps = 0

    def reset(self, seed=None, options=None):
        self.steps = 0
        goal = 1.0 if options is None else float(options["task_id"])
        return np.array([0.0]), {"goal": np.array([goal])}

    def step(self, action):
        self.steps += 1
        observation = np.array([float(self.steps)])
        success = float(action[0] > 0.0)
        return observation, success, bool(success), False, {"success": success}


class PositivePolicy:
    def __init__(self) -> None:
        self.reset_count = 0

    def reset_policy(self) -> None:
        self.reset_count += 1

    def predict(self, observations, goals=None):
        return np.ones((observations.shape[0], 1))


def test_goal_conditioned_rollout_evaluation_reports_success() -> None:
    policy = PositivePolicy()
    summary = evaluate_goal_conditioned_policy(FakeGoalEnv(), policy, episodes=3, seed=0, max_steps=5)

    assert summary.episodes == 3
    assert summary.success_rate == 1.0
    assert summary.return_mean == 1.0
    assert summary.length_mean == 1.0
    assert policy.reset_count == 3


class HorizonPolicy(PositivePolicy):
    def predict_with_info(self, observations, goals=None):
        return np.ones((observations.shape[0], 1)), {
            "selected_horizon": np.full(observations.shape[0], 4),
            "horizon_candidates": np.array([1, 4]),
            "horizon_probabilities": np.tile(
                np.array([[0.25, 0.75]]),
                (observations.shape[0], 1),
            ),
        }


def test_rollout_evaluation_records_selected_horizons() -> None:
    summary = evaluate_goal_conditioned_policy(FakeGoalEnv(), HorizonPolicy(), episodes=2, seed=0)

    assert summary.selected_horizon_mean == 4.0
    assert summary.as_metrics()["rollout_selected_horizon_mean"] == 4.0


def test_rollout_evaluation_reports_fixed_task_success_rates() -> None:
    summary = evaluate_goal_conditioned_policy(
        FakeGoalEnv(),
        PositivePolicy(),
        episodes=2,
        seed=0,
        task_ids=(1, 2),
    )

    assert summary.episodes == 4
    assert summary.task_success_rates == {1: 1.0, 2: 1.0}
    assert summary.as_metrics()["rollout_task_2_success_rate"] == 1.0


def test_rollout_evaluation_reports_task_gate_probabilities() -> None:
    summary = evaluate_goal_conditioned_policy(
        FakeGoalEnv(),
        HorizonPolicy(),
        episodes=2,
        seed=0,
        task_ids=(1, 2),
    )

    assert summary.task_selected_horizon_means == {1: 4.0, 2: 4.0}
    assert summary.task_horizon_probability_means == {
        1: {1: 0.25, 4: 0.75},
        2: {1: 0.25, 4: 0.75},
    }
    metrics = summary.as_metrics()
    assert metrics["rollout_task_1_selected_horizon_mean"] == 4.0
    assert metrics["rollout_task_2_horizon_4_probability"] == 0.75
