"""共用 pytest fixtures，提供最小化测试用 registry / schema 数据。"""
from __future__ import annotations

import textwrap
from pathlib import Path

import pytest


@pytest.fixture
def minimal_reference_catalog(tmp_path: Path) -> Path:
    """返回只含 2 个数据集的小型 catalog 文件路径。"""
    p = tmp_path / "reference_catalog.yaml"
    p.write_text(textwrap.dedent("""
        GLEAM_v4.2a_LowRes:
          name: GLEAM_v4.2a_LowRes
          category: Water
          data_type: grid
          grid_res: 0.5
          tim_res: Month
          description: GLEAM v4.2a LowRes ET dataset
          variables:
            Evapotranspiration:
              varname: E
              varunit: mm/day
            Transpiration:
              varname: Et
              varunit: mm/day
        FLUXNET_PLUMBER2:
          name: FLUXNET_PLUMBER2
          category: Water
          data_type: stn
          tim_res: Hour
          description: FLUXNET PLUMBER2 station data
          variables:
            Evapotranspiration:
              varname: ET
              varunit: kg/m2/s
    """).strip())
    return p


@pytest.fixture
def minimal_model_catalog(tmp_path: Path) -> Path:
    p = tmp_path / "model_catalog.yaml"
    p.write_text(textwrap.dedent("""
        CoLM2024:
          name: CoLM2024
          data_type: grid
          tim_res: Month
          description: Common Land Model 2024
          variables:
            Evapotranspiration:
              varname: f_lh_vap
              varunit: W/m2
            Latent_Heat:
              varname: f_lh
              varunit: W/m2
    """).strip())
    return p
