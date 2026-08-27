from scripts.plot_summary_bars import svg_bar_chart


def test_svg_bar_chart_contains_algorithm_labels() -> None:
    svg = svg_bar_chart(
        {"by_algorithm": {"iql": {"mean": 0.4}, "adaptive_iql": {"mean": 0.3}}},
        title="Pilot",
        ylabel="MSE",
        lower_is_better=True,
    )

    assert svg.startswith("<svg")
    assert "Pilot" in svg
    assert "iql" in svg
    assert "adaptive_iql" in svg
    assert "lower is better" in svg


def test_svg_bar_chart_supports_group_key() -> None:
    svg = svg_bar_chart(
        {"by_algorithm_chunk": {"iql_chunk_1": {"mean": 0.4}, "iql_chunk_4": {"mean": 0.3}}},
        title="Chunks",
        ylabel="Success",
        group_key="by_algorithm_chunk",
    )

    assert "iql_chunk_1" in svg
    assert "iql_chunk_4" in svg
