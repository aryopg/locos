import importlib
import importlib.resources as resources
import json
from pathlib import Path

import locos_eval


def test_locos_eval_public_api_excludes_decore_decoding():
    forbidden = {"DeCoreLLM", "DeCoreWrapper", "DeCoreRPCWrapper", "decore"}
    assert forbidden.isdisjoint(set(locos_eval.__all__))
    for name in forbidden:
        assert not hasattr(locos_eval, name)


def test_migrated_revision_modules_importable():
    modules = [
        "locos.analysis._utils",
        "locos.analysis.e1_alpha_spatial",
        "locos.analysis.e2_topk_overlap",
        "locos.analysis.e3_provenance",
        "locos.analysis.e4_nonliterality",
        "locos.analysis.e5_benchmark_literality",
        "locos.analysis.e6_babilong_errors",
        "locos.analysis.e7_factorial",
        "locos.analysis.e7_gemma27b_lens",
        "locos.analysis.inventory",
        "locos.detectors.dla",
        "locos.plotting._downstream_bar_common",
        "locos.plotting.babilong_bar",
        "locos.plotting.musique_bar",
    ]
    for module in modules:
        importlib.import_module(module)


def test_demo_notebook_is_valid_json_and_has_gpu_markers():
    notebook = Path("notebooks/locos_demo.ipynb")
    data = json.loads(notebook.read_text())
    assert data["nbformat"] == 4
    source = "\n".join("".join(cell.get("source", [])) for cell in data["cells"])
    assert "GPU-required detection" in source
    assert "make detect" in source
    assert "make ablate" in source


def test_prompt_and_model_yaml_resources_are_packaged():
    assert (resources.files("locos.analysis") / "prompts" / "parametric_ablation.yaml").is_file()
    assert (resources.files("locos_eval.evals") / "configs" / "_default.yaml").is_file()
    assert (resources.files("locos_eval.evals") / "prompts" / "babilong.yaml").is_file()
