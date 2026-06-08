"""Gridded-data calculations for Diff Plot comparison."""

from __future__ import annotations

import logging
import os

import xarray as xr

from openbench.util.converttype import Convert_Type
from openbench.util.filenames import diff_grid_anomaly_filename, diff_grid_difference_filename
from openbench.util.netcdf import write_netcdf_atomic as _write_netcdf_atomic


def process_grid_diff_plot(
    *,
    basedir: str,
    dir_path: str,
    evaluation_item: str,
    ref_source: str,
    sim_sources: list[str],
    metrics: list[str],
    scores: list[str],
) -> None:
    # Calculate ensemble means and anomalies for metrics
    for metric in metrics:
        try:
            # Load all simulation data for this metric
            datasets = []
            for sim_source in sim_sources:
                with xr.open_dataset(
                    os.path.join(
                        basedir,
                        "metrics",
                        f"{evaluation_item}_ref_{ref_source}_sim_{sim_source}_{metric}.nc",
                    )
                ) as ds_file:
                    ds = Convert_Type.convert_nc(ds_file.load())
                datasets.append(ds[metric])
            # Calculate ensemble mean
            ensemble_mean = xr.concat(datasets, dim="ensemble").mean("ensemble")

            # Save ensemble mean
            ds_mean = xr.Dataset()
            ds_mean[f"{metric}_ensemble_mean"] = Convert_Type.convert_nc(ensemble_mean)
            ds_mean.attrs["description"] = f"Ensemble mean of {metric} across all simulations"
            output_file = os.path.join(dir_path, f"{evaluation_item}_ref_{ref_source}_ensemble_mean_{metric}.nc")
            ds_mean = Convert_Type.convert_nc(ds_mean)
            _write_netcdf_atomic(ds_mean, output_file)

            # Calculate and save anomalies for each simulation
            for sim_source, ds in zip(sim_sources, datasets):
                anomaly = ds - ensemble_mean
                ds_anom = xr.Dataset()
                ds_anom[f"{metric}_anomaly"] = anomaly
                ds_anom.attrs["description"] = f"Anomaly from ensemble mean for {sim_source}"
                output_file = os.path.join(
                    dir_path, diff_grid_anomaly_filename(evaluation_item, ref_source, sim_source, metric)
                )
                ds_anom = Convert_Type.convert_nc(ds_anom)
                _write_netcdf_atomic(ds_anom, output_file)

        except Exception as e:
            logging.error(f"Error processing ensemble calculations for metric {metric}: {e}")
            raise

    # Calculate ensemble means and anomalies for scores
    for score in scores:
        try:
            # Load all simulation data for this score
            datasets = []
            for sim_source in sim_sources:
                with xr.open_dataset(
                    os.path.join(
                        basedir,
                        "scores",
                        f"{evaluation_item}_ref_{ref_source}_sim_{sim_source}_{score}.nc",
                    )
                ) as ds_file:
                    ds = Convert_Type.convert_nc(ds_file.load())
                datasets.append(ds[score])

            # Calculate ensemble mean
            ensemble_mean = xr.concat(datasets, dim="ensemble").mean("ensemble")

            # Save ensemble mean
            ds_mean = xr.Dataset()
            ds_mean[f"{score}_ensemble_mean"] = ensemble_mean
            ds_mean.attrs["description"] = f"Ensemble mean of {score} across all simulations"
            output_file = os.path.join(dir_path, f"{evaluation_item}_ref_{ref_source}_ensemble_mean_{score}.nc")
            ds_mean = Convert_Type.convert_nc(ds_mean)
            _write_netcdf_atomic(ds_mean, output_file)

            # Calculate and save anomalies for each simulation
            for sim_source, ds in zip(sim_sources, datasets):
                anomaly = ds - ensemble_mean
                ds_anom = xr.Dataset()
                ds_anom[f"{score}_anomaly"] = anomaly
                ds_anom.attrs["description"] = f"Anomaly from ensemble mean for {sim_source}"
                output_file = os.path.join(
                    dir_path, diff_grid_anomaly_filename(evaluation_item, ref_source, sim_source, score)
                )
                ds_anom = Convert_Type.convert_nc(ds_anom)
                _write_netcdf_atomic(ds_anom, output_file)

        except Exception as e:
            logging.error(f"Error processing ensemble calculations for score {score}: {e}")
            raise
    if len(sim_sources) >= 2:
        # Compare metrics between pairs
        for metric in metrics:
            for i, sim1 in enumerate(sim_sources):
                for j, sim2 in enumerate(sim_sources[i + 1 :], i + 1):
                    try:
                        with xr.open_dataset(
                            os.path.join(
                                basedir,
                                "metrics",
                                f"{evaluation_item}_ref_{ref_source}_sim_{sim1}_{metric}.nc",
                            )
                        ) as ds1_file:
                            ds1 = Convert_Type.convert_nc(ds1_file.load())
                        with xr.open_dataset(
                            os.path.join(
                                basedir,
                                "metrics",
                                f"{evaluation_item}_ref_{ref_source}_sim_{sim2}_{metric}.nc",
                            )
                        ) as ds2_file:
                            ds2 = Convert_Type.convert_nc(ds2_file.load())
                        diff = ds1[metric] - ds2[metric]

                        ds_out = xr.Dataset()
                        ds_out[f"{metric}_diff"] = diff
                        ds_out.attrs["description"] = f"Difference in {metric} between {sim1} and {sim2}"

                        output_file = os.path.join(
                            dir_path,
                            diff_grid_difference_filename(evaluation_item, ref_source, sim1, sim2, metric),
                        )
                        ds_out = Convert_Type.convert_nc(ds_out)
                        _write_netcdf_atomic(ds_out, output_file)

                    except Exception as e:
                        logging.error(f"Error processing metric {metric} for {sim1} vs {sim2}: {e}")
                        raise

        # Compare scores between pairs
        for score in scores:
            for i, sim1 in enumerate(sim_sources):
                for j, sim2 in enumerate(sim_sources[i + 1 :], i + 1):
                    try:
                        with xr.open_dataset(
                            os.path.join(
                                basedir,
                                "scores",
                                f"{evaluation_item}_ref_{ref_source}_sim_{sim1}_{score}.nc",
                            )
                        ) as ds1_file:
                            ds1 = Convert_Type.convert_nc(ds1_file.load())
                        with xr.open_dataset(
                            os.path.join(
                                basedir,
                                "scores",
                                f"{evaluation_item}_ref_{ref_source}_sim_{sim2}_{score}.nc",
                            )
                        ) as ds2_file:
                            ds2 = Convert_Type.convert_nc(ds2_file.load())
                        diff = ds1[score] - ds2[score]

                        ds_out = xr.Dataset()
                        ds_out[f"{score}_diff"] = diff
                        ds_out.attrs["description"] = f"Difference in {score} between {sim1} and {sim2}"

                        output_file = os.path.join(
                            dir_path,
                            diff_grid_difference_filename(evaluation_item, ref_source, sim1, sim2, score),
                        )
                        ds_out = Convert_Type.convert_nc(ds_out)
                        _write_netcdf_atomic(ds_out, output_file)

                    except Exception as e:
                        logging.error(f"Error processing score {score} for {sim1} vs {sim2}: {e}")
                        raise
