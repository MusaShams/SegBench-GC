from scripts.analyze_segmentation_robustness import summarize


def test_segmentation_summary_reports_sensitivity_ratios() -> None:
    payload = {
        "original": [
            {"eval_action_mse": 1.0},
            {"eval_action_mse": 1.1},
        ]
    }
    for mode, values in {
        "robust": (1.0, 1.1, 0.9),
        "naive": (1.0, 2.0, 0.0),
    }.items():
        for suffix, value in zip(
            ("_offset_4", "_offset_14", ""),
            values,
        ):
            payload[f"{mode}_cut_25{suffix}"] = [
                {"eval_action_mse": value},
                {"eval_action_mse": value + 0.05},
            ]

    summary = summarize(payload)

    assert summary["variance_ratio_naive_over_robust"] > 1.0
    assert (
        summary["offset_range_ratio_naive_over_robust"] > 1.0
    )
