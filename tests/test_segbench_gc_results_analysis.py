import json

from scripts.analyze_segbench_gc_results import (
    bootstrap_mean_ci,
    condition_mode,
    read_final_events,
    summarize_broader_runs,
    summarize_manipulation_runs,
    update_step_provenance,
    write_key_seed_table,
)


def test_condition_mode_groups_offsets() -> None:
    assert condition_mode("original") == "original"
    assert condition_mode("robust-offset24") == "robust"
    assert condition_mode("naive-offset14") == "naive"


def test_bootstrap_mean_ci_is_deterministic() -> None:
    first = bootstrap_mean_ci([0.1, 0.2, 0.3], samples=100, seed=4)
    second = bootstrap_mean_ci([0.1, 0.2, 0.3], samples=100, seed=4)

    assert first == second


def test_read_final_events_requires_complete_log(tmp_path) -> None:
    path = tmp_path / "metrics.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "event": "train_start",
                        "config": {"task": "test", "steps": 1},
                    }
                ),
                json.dumps(
                    {
                        "event": "eval",
                        "metrics": {"eval_action_mse": 0.5},
                    }
                ),
            ]
        ),
        encoding="utf-8",
    )

    try:
        read_final_events(path)
    except ValueError as error:
        assert "Incomplete run log" in str(error)
    else:
        raise AssertionError("Incomplete logs must be rejected.")


def test_read_final_events_preserves_total_update_step(tmp_path) -> None:
    path = tmp_path / "metrics.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "event": "train_start",
                        "config": {"task": "test", "steps": 900_000},
                    }
                ),
                json.dumps(
                    {
                        "event": "eval",
                        "step": 1_000_000,
                        "metrics": {"eval_action_mse": 0.5},
                    }
                ),
                json.dumps(
                    {
                        "event": "rollout_eval",
                        "step": 1_000_000,
                        "metrics": {"rollout_success_rate": 0.2},
                    }
                ),
            ]
        ),
        encoding="utf-8",
    )

    _, _, rollout = read_final_events(path)

    assert rollout["_event_step"] == 1_000_000


def test_update_step_provenance_labels_resumed_run() -> None:
    provenance = update_step_provenance(
        {"config": {"steps": 900_000}},
        {"_event_step": 1_000_000.0},
    )

    assert provenance == {
        "initial_update_steps": 100_000,
        "invocation_update_steps": 900_000,
        "resumed_update_steps": 900_000,
        "total_update_steps": 1_000_000,
        "resumed_from_checkpoint": True,
    }


def test_broader_summary_keeps_studies_separate() -> None:
    runs = []
    for study, modes in {
        "fixed_h8": ("original", "robust", "naive"),
        "random_cuts": ("robust", "naive"),
        "random_cuts_1m": ("robust", "naive"),
        "fixed_h8_1m": ("original", "robust", "naive"),
    }.items():
        for mode in modes:
            for seed in range(2):
                runs.append(
                    {
                        "study": study,
                        "mode": mode,
                        "seed": seed,
                        "backup_boundary_count": 10,
                        "success_rate": 0.5,
                        "action_mse": 1.0,
                    }
                )

    summary = summarize_broader_runs(runs)

    assert len(summary) == 10
    assert {
        (row["study"], row["mode"])
        for row in summary
    } == {
        ("fixed_h8", "original"),
        ("fixed_h8", "robust"),
        ("fixed_h8", "naive"),
        ("random_cuts", "robust"),
        ("random_cuts", "naive"),
        ("random_cuts_1m", "robust"),
        ("random_cuts_1m", "naive"),
        ("fixed_h8_1m", "original"),
        ("fixed_h8_1m", "robust"),
        ("fixed_h8_1m", "naive"),
    }


def test_manipulation_summary_handles_single_seed() -> None:
    runs = []
    for study, seeds in {
        "cube_double_10k": range(2),
        "cube_double_100k": range(2),
        "cube_double_1m_seed0": range(1),
    }.items():
        for mode in ("original", "robust", "naive"):
            for seed in seeds:
                runs.append(
                    {
                        "study": study,
                        "mode": mode,
                        "seed": seed,
                        "success_rate": 0.0,
                        "action_mse": 1.0,
                    }
                )

    summary = summarize_manipulation_runs(runs)
    full_rows = [
        row
        for row in summary
        if row["study"] == "cube_double_1m_seed0"
    ]

    assert len(summary) == 9
    assert all(row["success_sample_std"] == "" for row in full_rows)


def test_key_seed_table_is_generated_from_run_rows(tmp_path) -> None:
    runs = []
    for experiment, seeds in {
        "medium_1m": range(5),
        "antmaze_1m": range(3),
    }.items():
        for mode in ("original", "robust", "naive"):
            for seed in seeds:
                runs.append(
                    {
                        "experiment": experiment,
                        "mode": mode,
                        "seed": seed,
                        "success_rate": 0.1,
                    }
                )
    broader = []
    for study in ("random_cuts_1m", "fixed_h8_1m"):
        modes = (
            ("robust", "naive")
            if study == "random_cuts_1m"
            else ("original", "robust", "naive")
        )
        for mode in modes:
            for seed in range(3):
                broader.append(
                    {
                        "study": study,
                        "mode": mode,
                        "seed": seed,
                        "success_rate": 0.2,
                    }
                )
    output = tmp_path / "seed-results.tex"

    write_key_seed_table(output, runs, broader)

    text = output.read_text(encoding="utf-8")
    assert "PointMaze multi-head periodic" in text
    assert "AntMaze multi-head periodic" in text
    assert "20.0" in text
