"""Independent Overall Score completeness artifacts; never modify scoring inputs."""

import json
import logging
from pathlib import Path

import numpy as np
import xarray as xr

NAMES = ("nBiasScore", "nRMSEScore", "nPhaseScore", "nIavScore", "nSpatialScore")
WEIGHTS = (1, 2, 1, 1, 1)


def write_diagnostics(evaluator, components, overall, paired, *, stations=None, vkey=""):
    arrays = xr.align(*components, overall, paired, join="exact")
    arrays = xr.broadcast(*arrays)
    fields, overall, paired = arrays[:5], arrays[5], arrays[6]
    valid = [np.isfinite(a) for a in fields]
    count = sum(a.astype("int8") for a in valid)
    weight = sum(a.astype("int8") * w for a, w in zip(valid, WEIGHTS))
    spatial_only = valid[4] & (count == 1)
    finite = np.isfinite(overall)
    total = count.size
    summary = {
        "item": evaluator.item,
        "reference": evaluator.ref_source,
        "simulation": evaluator.sim_source,
        "total_locations": total,
        "paired_data_locations": int(paired.sum()),
        "domain_definition": "All supplied evaluation coordinates; missing pairs do not imply outside-domain.",
        "fraction_denominator": "total_locations",
        "valid_component_counts": {
            str(k): {"count": int((count == k).sum()), "fraction": float((count == k).sum()) / total if total else 0.0}
            for k in range(6)
        },
        "components": {name: {"valid": int(a.sum()), "missing": total - int(a.sum())} for name, a in zip(NAMES, valid)},
        "spatial_only": int(spatial_only.sum()),
        "any_nonspatial": int((count - valid[4].astype("int8") > 0).sum()),
        "finite_overall": int(finite.sum()),
        "finite_overall_incomplete": int((finite & (count < 5)).sum()),
    }
    # Per-task directories avoid collisions and keep fixed artifact names.
    target = (
        Path(evaluator.casedir)
        / "diagnostics"
        / "overall_score"
        / (f"{evaluator.item}__ref__{evaluator.ref_source}__sim__{evaluator.sim_source}{vkey}")
    )
    base = target
    run = 0
    while True:
        try:
            target.mkdir(parents=True, exist_ok=False)
            break
        except FileExistsError:
            run += 1
            target = base.with_name(f"{base.name}__run_{run}")
    dataset = xr.Dataset(
        {
            "valid_component_count": count,
            "valid_weight_sum": weight,
            "all_five_valid_mask": (count == 5).astype("int8"),
            "paired_data_mask": paired.astype("int8"),
        }
    )
    for name, a in zip(NAMES, valid):
        dataset[name + "_valid_mask"] = a.astype("int8")
    for name in dataset:
        if name.endswith("mask"):
            dataset[name].attrs.update(
                flag_values=np.array([-1, 0, 1], dtype="int8"),
                flag_meanings="outside_domain invalid valid",
                comment="All supplied coordinates are in-domain; -1 reserved for explicit domain exclusion.",
            )
    dataset.valid_component_count.attrs.update(valid_min=0, valid_max=5)
    dataset.valid_weight_sum.attrs.update(valid_min=0, valid_max=6, comment="Weights 1,2,1,1,1, not component count")
    if stations is None:
        dataset.to_netcdf(target / "component_validity.nc")
    else:
        frame = dataset.to_dataframe().reset_index(drop=True)
        frame.insert(0, "ID", np.asarray(stations))
        frame["Overall_Score"] = overall.values
        frame.to_csv(target / "component_validity.csv", index=False)
    (target / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logging.info(
        "Overall_Score completeness: %s/%s all-five valid; %s spatial-only; details: %s",
        summary["valid_component_counts"]["5"]["count"],
        total,
        summary["spatial_only"],
        target,
    )
    if summary["finite_overall_incomplete"]:
        logging.warning(
            "Overall_Score component completeness: counts 0..5=%s; spatial-only=%s; "
            "finite Overall_Score with incomplete components=%s. These values use renormalized weights. See %s",
            summary["valid_component_counts"],
            summary["spatial_only"],
            summary["finite_overall_incomplete"],
            target,
        )
    return summary
