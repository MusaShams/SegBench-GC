from adaptive_gcrl.envs.benchmark_setup import module_status, ogbench_setup_status


def test_module_status_reports_importability() -> None:
    status = module_status("json")

    assert status.name == "json"
    assert status.installed is True


def test_ogbench_setup_status_has_expected_fields() -> None:
    status = ogbench_setup_status()

    assert "dependencies" in status
    assert "ready" in status
    assert "notes" in status

