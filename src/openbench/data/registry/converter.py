"""Convert old-format reference variable definition files to new registry format."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import yaml


def convert_old_reference(
    old_path: Path,
    output_path: Path,
    name: str,
    category: str = "",
    description: str = "",
) -> None:
    """Convert a single old-format reference definition to new registry format."""
    with open(old_path) as f:
        old = yaml.safe_load(f)

    general = old.get("general", {})

    variables: dict[str, Any] = {}
    for key, value in old.items():
        if key != "general" and isinstance(value, dict):
            var_entry: dict[str, Any] = {"varname": value.get("varname", "")}
            if value.get("varunit"):
                var_entry["varunit"] = value["varunit"]
            if value.get("prefix"):
                var_entry["prefix"] = value["prefix"]
            if value.get("suffix"):
                var_entry["suffix"] = value["suffix"]
            if value.get("sub_dir"):
                var_entry["sub_dir"] = value["sub_dir"]
            if value.get("fulllist"):
                var_entry["fulllist"] = value["fulllist"]
            if value.get("max_uparea"):
                var_entry["max_uparea"] = value["max_uparea"]
            if value.get("min_uparea"):
                var_entry["min_uparea"] = value["min_uparea"]
            variables[key] = var_entry

    new_descriptor: dict[str, Any] = {
        "name": name,
        "description": description or f"{name} reference dataset",
        "category": category,
        "data_type": general.get("data_type", "grid"),
        "tim_res": general.get("tim_res", "Month"),
        "data_groupby": general.get("data_groupby", "Year"),
        "timezone": general.get("timezone", 0),
        "years": [general.get("syear", 2000), general.get("eyear", 2020)],
        "variables": variables,
    }

    if general.get("grid_res"):
        new_descriptor["grid_res"] = general["grid_res"]
    if general.get("fulllist"):
        new_descriptor["fulllist"] = general["fulllist"]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        yaml.dump(new_descriptor, f, default_flow_style=False, sort_keys=False, allow_unicode=True)


def batch_convert_references(
    source_dir: Path, output_dir: Path, category_map: Optional[dict[str, str]] = None
) -> list[str]:
    """Convert all old-format reference definitions in a directory."""
    if category_map is None:
        category_map = {}

    converted = []
    for old_file in sorted(source_dir.glob("*.yaml")):
        name = old_file.stem
        category = category_map.get(name, _guess_category(old_file))
        output_path = output_dir / f"{name}.yaml"
        try:
            convert_old_reference(old_file, output_path, name=name, category=category)
            converted.append(name)
        except Exception as e:
            print(f"Warning: Failed to convert {old_file.name}: {e}")

    return converted


def _guess_category(path: Path) -> str:
    """Guess dataset category from its variable names."""
    with open(path) as f:
        data = yaml.safe_load(f)

    variables = [k for k in data.keys() if k != "general"]

    water_vars = {
        "Evapotranspiration",
        "Runoff",
        "Total_Runoff",
        "Streamflow",
        "Soil_Moisture",
        "Surface_Soil_Moisture",
        "Root_Zone_Soil_Moisture",
        "Snow_Depth",
        "Snow_Water_Equivalent",
        "Precipitation",
        "Canopy_Interception",
        "Canopy_Transpiration",
        "Bare_Soil_Evaporation",
        "Terrestrial_Water_Storage_Change",
        "Water_Table_Depth",
        "Groundwater_Recharge_Rate",
        "Inundation_Fraction",
        "Inundation_Area",
        "Depth_Of_Surface_Water",
        "Open_Water_Evaporation",
    }
    carbon_vars = {
        "Gross_Primary_Productivity",
        "Net_Ecosystem_Exchange",
        "Ecosystem_Respiration",
        "Biomass",
        "Leaf_Area_Index",
        "Soil_Carbon",
        "Net_Primary_Production",
        "Veg_Cover_In_Fraction",
        "Burned_Area",
        "Methane",
        "Wetland_Methane_Emission",
        "Leaf_Greenness",
    }
    energy_vars = {
        "Latent_Heat",
        "Sensible_Heat",
        "Net_Radiation",
        "Ground_Heat",
        "Surface_Upward_SW_Radiation",
        "Surface_Upward_LW_Radiation",
        "Surface_Net_SW_Radiation",
        "Surface_Net_LW_Radiation",
        "Surface_Downward_SW_Radiation",
        "Surface_Downward_LW_Radiation",
        "Surface_Albedo",
    }

    for v in variables:
        if v in water_vars:
            return "Water"
        if v in carbon_vars:
            return "Carbon"
        if v in energy_vars:
            return "Energy"

    return "Other"
