# tests/test_config_loader.py
import tempfile, os
from pathlib import Path
from src.utils.config_loader import ConfigLoader


def test_load_global_config(tmp_path):
    global_yaml = tmp_path / "global.yaml"
    global_yaml.write_text("paths:\n  data_root: /tmp/test\n  output_dir: ./out\n")
    loader = ConfigLoader(tmp_path)
    cfg = loader.load_global()
    assert cfg["paths"]["data_root"] == "/tmp/test"


def test_load_dataset_config(tmp_path):
    ds_yaml = tmp_path / "test.yaml"
    ds_yaml.write_text("dataset:\n  name: TestDS\nsource:\n  reader: test\n")
    loader = ConfigLoader(tmp_path)
    cfg = loader.load_dataset(ds_yaml)
    assert cfg["dataset"]["name"] == "TestDS"


def test_env_var_substitution(tmp_path):
    global_yaml = tmp_path / "global.yaml"
    global_yaml.write_text("paths:\n  data_root: ${TEST_SF_ROOT:/default/path}\n")
    loader = ConfigLoader(tmp_path)
    cfg = loader.load_global()
    assert cfg["paths"]["data_root"] == "/default/path"

    os.environ["TEST_SF_ROOT"] = "/env/path"
    cfg = loader.load_global()
    assert cfg["paths"]["data_root"] == "/env/path"
    del os.environ["TEST_SF_ROOT"]


def test_load_all_dataset_configs(tmp_path):
    ds_dir = tmp_path / "datasets"
    ds_dir.mkdir()
    (ds_dir / "a.yaml").write_text("dataset:\n  name: A\nsource:\n  reader: a\n")
    (ds_dir / "b.yaml").write_text("dataset:\n  name: B\nsource:\n  reader: b\n")
    (ds_dir / "_template.yaml").write_text("# template\n")
    loader = ConfigLoader(tmp_path)
    configs = loader.load_all_datasets(ds_dir)
    assert len(configs) == 2  # _template excluded
    names = {c["dataset"]["name"] for c in configs}
    assert names == {"A", "B"}
