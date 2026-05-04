from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


def _load_export_module():
    script = Path(__file__).resolve().parents[1] / "scripts" / "export_paper_results.py"
    spec = spec_from_file_location("export_paper_results", script)
    assert spec is not None
    assert spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_collect_artifacts_includes_single_task_inference_files(tmp_path: Path):
    module = _load_export_module()
    for name in (
        "summary.csv",
        "inference_time.csv",
        "inference_clusters.csv",
        "inference_logistic_time.csv",
        "inference_logistic_clusters.csv",
        "sub-01_time_decode.csv",
    ):
        (tmp_path / name).write_text("x\n", encoding="utf-8")

    artifacts = module.collect_artifacts(tmp_path, module.DEFAULT_PATTERNS)

    assert {path.name for path in artifacts} == {
        "summary.csv",
        "inference_time.csv",
        "inference_clusters.csv",
        "inference_logistic_time.csv",
        "inference_logistic_clusters.csv",
    }
