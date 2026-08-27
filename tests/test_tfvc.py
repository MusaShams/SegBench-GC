from adaptive_gcrl.utils.tfvc import current_tfvc_changeset, tf_executable


def test_tf_executable_resolution_is_optional() -> None:
    executable = tf_executable()

    assert executable is None or executable.endswith("/tf")


def test_tf_executable_supports_explicit_configuration(
    monkeypatch,
) -> None:
    monkeypatch.setenv("TFVC_CLI", "/portable/tools/tf")

    assert tf_executable() == "/portable/tools/tf"


def test_tfvc_metadata_is_optional_without_workspace() -> None:
    assert current_tfvc_changeset("", "") is None
