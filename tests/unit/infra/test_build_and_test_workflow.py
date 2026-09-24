from pathlib import Path

import yaml


WORKFLOW = Path(__file__).parents[3] / ".github" / "workflows" / "build_and_test.yml"


def _step(job: dict, name: str) -> dict:
    return next(step for step in job["steps"] if step.get("name") == name)


def test_sdk_junit_publication_survives_linux_test_failures() -> None:
    workflow = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    job = workflow["jobs"]["main-job"]

    publish = _step(job, "Publish Pytest Results")
    assert publish["if"] == "runner.os == 'Linux' && (success() || failure())"
    assert publish["with"]["files"] == "**/test-*.xml"

    upload = _step(job, "Upload SDK JUnit XML")
    assert upload["if"] == publish["if"]
    assert upload["uses"].startswith("actions/upload-artifact@")
    assert upload["with"]["if-no-files-found"] == "warn"
    assert "matrix.os" in upload["with"]["name"]
    assert "matrix.python" in upload["with"]["name"]
    assert "matrix.package_extras" in upload["with"]["name"]
