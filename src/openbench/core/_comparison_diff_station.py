"""Station-data calculations for Diff Plot comparison."""

from __future__ import annotations

import logging
import os

import pandas as pd

from openbench.core._comparison_helpers import (
    _station_frames_aligned_by_id,
    _station_pairwise_difference_by_id,
    _write_csv_atomic,
)
from openbench.util.converttype import Convert_Type
from openbench.util.filenames import diff_station_anomaly_filename, diff_station_difference_filename


def process_station_diff_plot(
    *,
    basedir: str,
    dir_path: str,
    sim_nml: dict,
    evaluation_item: str,
    ref_source: str,
    sim_sources: list[str],
    metrics: list[str],
    scores: list[str],
) -> None:
    # Process metrics for station data
    for metric in metrics:
        try:
            # Load all station data for this metric
            station_frames = {}
            for sim_source in sim_sources:
                sim_nml[f"{evaluation_item}"][f"{sim_source}_varname"]
                file_path = os.path.join(
                    basedir,
                    "metrics",
                    f"{evaluation_item}_stn_{ref_source}_{sim_source}_evaluations.csv",
                )
                df = pd.read_csv(file_path, sep=",", header=0)
                df = Convert_Type.convert_Frame(df)
                station_frames[sim_source] = df

            # Convert to DataFrame for easier handling
            aligned_frames = _station_frames_aligned_by_id(station_frames, metric)
            station_df = pd.DataFrame(
                {sim_source: aligned_frames[sim_source][metric].reset_index(drop=True) for sim_source in sim_sources}
            )
            base_df = aligned_frames[sim_sources[0]]

            # Calculate ensemble mean
            ensemble_mean = station_df.mean(axis=1).astype("float32")
            ensemble_df = pd.DataFrame({"ID": base_df["ID"], f"{metric}_ensemble_mean": ensemble_mean})
            ensemble_df = Convert_Type.convert_Frame(ensemble_df)
            _write_csv_atomic(
                ensemble_df,
                os.path.join(dir_path, f"{evaluation_item}_stn_{ref_source}_ensemble_mean_{metric}.csv"),
                index=False,
            )

            # Calculate anomalies for each simulation
            for sim_source in sim_sources:
                df = aligned_frames[sim_source]
                try:
                    lon_select = df["ref_lon"].values
                    lat_select = df["ref_lat"].values
                except (KeyError, ValueError) as e:
                    logging.debug(f"Using sim coordinates instead of ref coordinates: {e}")
                    lon_select = df["sim_lon"].values
                    lat_select = df["sim_lat"].values
                anomaly = station_df[sim_source] - ensemble_mean
                anomaly_df = pd.DataFrame(
                    {"ID": df["ID"], "lat": lat_select, "lon": lon_select, f"{metric}_anomaly": anomaly}
                )
                anomaly_df = Convert_Type.convert_Frame(anomaly_df)
                _write_csv_atomic(
                    anomaly_df,
                    os.path.join(
                        dir_path,
                        diff_station_anomaly_filename(evaluation_item, ref_source, sim_source, metric),
                    ),
                    index=False,
                )

        except Exception as e:
            logging.error(f"Error processing station ensemble calculations for metric {metric}: {e}")
            raise

    # Process scores for station data
    for score in scores:
        try:
            # Load all station data for this score
            station_frames = {}
            for sim_source in sim_sources:
                file_path = f"{basedir}/scores/{evaluation_item}_stn_{ref_source}_{sim_source}_evaluations.csv"
                df = pd.read_csv(file_path, sep=",", header=0)
                df = Convert_Type.convert_Frame(df)
                station_frames[sim_source] = df

            # Convert to DataFrame for easier handling
            aligned_frames = _station_frames_aligned_by_id(station_frames, score)
            station_df = pd.DataFrame(
                {sim_source: aligned_frames[sim_source][score].reset_index(drop=True) for sim_source in sim_sources}
            )
            base_df = aligned_frames[sim_sources[0]]

            # Calculate ensemble mean
            ensemble_mean = station_df.mean(axis=1).astype("float32")
            ensemble_df = pd.DataFrame({"ID": base_df["ID"], f"{score}_ensemble_mean": ensemble_mean})
            ensemble_df = Convert_Type.convert_Frame(ensemble_df)
            _write_csv_atomic(
                ensemble_df,
                os.path.join(dir_path, f"{evaluation_item}_stn_{ref_source}_ensemble_mean_{score}.csv"),
                index=False,
            )

            # Calculate anomalies for each simulation
            for sim_source in sim_sources:
                df = aligned_frames[sim_source]
                try:
                    lon_select = df["ref_lon"].values
                    lat_select = df["ref_lat"].values
                except (KeyError, ValueError) as e:
                    logging.debug(f"Using sim coordinates instead of ref coordinates: {e}")
                    lon_select = df["sim_lon"].values
                    lat_select = df["sim_lat"].values
                anomaly = station_df[sim_source] - ensemble_mean
                anomaly_df = pd.DataFrame(
                    {"ID": df["ID"], "lat": lat_select, "lon": lon_select, f"{score}_anomaly": anomaly}
                )
                anomaly_df = Convert_Type.convert_Frame(anomaly_df)
                _write_csv_atomic(
                    anomaly_df,
                    os.path.join(
                        dir_path,
                        diff_station_anomaly_filename(evaluation_item, ref_source, sim_source, score),
                    ),
                    index=False,
                )

        except Exception as e:
            logging.error(f"Error processing station ensemble calculations for score {score}: {e}")
            raise
    if len(sim_sources) >= 2:
        # Calculate pairwise differences for metrics (station data)
        for metric in metrics:
            for i, sim1 in enumerate(sim_sources):
                sim_varname_1 = sim_nml[f"{evaluation_item}"][f"{sim1}_varname"]
                for j, sim2 in enumerate(sim_sources[i + 1 :], i + 1):
                    sim_varname_2 = sim_nml[f"{evaluation_item}"][f"{sim2}_varname"]
                    try:
                        df1 = pd.read_csv(
                            os.path.join(
                                basedir,
                                "metrics",
                                f"{evaluation_item}_stn_{ref_source}_{sim1}_evaluations.csv",
                            )
                        )
                        df2 = pd.read_csv(
                            os.path.join(
                                basedir,
                                "metrics",
                                f"{evaluation_item}_stn_{ref_source}_{sim2}_evaluations.csv",
                            )
                        )
                        df1 = Convert_Type.convert_Frame(df1)
                        df2 = Convert_Type.convert_Frame(df2)

                        df1, diff = _station_pairwise_difference_by_id(
                            df1,
                            df2,
                            metric,
                            left_label=sim1,
                            right_label=sim2,
                        )
                        try:
                            lon_select = df1["ref_lon"].values
                            lat_select = df1["ref_lat"].values
                        except (KeyError, ValueError) as e:
                            logging.debug(f"Using sim coordinates instead of ref coordinates: {e}")
                            lon_select = df1["sim_lon"].values
                            lat_select = df1["sim_lat"].values
                        diff_df = pd.DataFrame(
                            {
                                "ID": df1["ID"],
                                "lat": lat_select,
                                "lon": lon_select,
                                f"{metric}_diff": diff,
                            }
                        )

                        output_file = os.path.join(
                            dir_path,
                            diff_station_difference_filename(
                                evaluation_item,
                                ref_source,
                                sim1,
                                sim_varname_1,
                                sim2,
                                sim_varname_2,
                                metric,
                            ),
                        )
                        diff_df = Convert_Type.convert_Frame(diff_df)
                        _write_csv_atomic(diff_df, output_file, index=False)

                    except Exception as e:
                        logging.error(f"Error processing station metric {metric} for {sim1} vs {sim2}: {e}")
                        raise

        # Calculate pairwise differences for scores (station data)
        for score in scores:
            for i, sim1 in enumerate(sim_sources):
                sim_varname_1 = sim_nml[f"{evaluation_item}"][f"{sim1}_varname"]
                for j, sim2 in enumerate(sim_sources[i + 1 :], i + 1):
                    sim_varname_2 = sim_nml[f"{evaluation_item}"][f"{sim2}_varname"]
                    try:
                        df1 = pd.read_csv(
                            os.path.join(
                                basedir,
                                "scores",
                                f"{evaluation_item}_stn_{ref_source}_{sim1}_evaluations.csv",
                            )
                        )
                        df2 = pd.read_csv(
                            os.path.join(
                                basedir,
                                "scores",
                                f"{evaluation_item}_stn_{ref_source}_{sim2}_evaluations.csv",
                            )
                        )
                        df1 = Convert_Type.convert_Frame(df1)
                        df2 = Convert_Type.convert_Frame(df2)
                        df1, diff = _station_pairwise_difference_by_id(
                            df1,
                            df2,
                            score,
                            left_label=sim1,
                            right_label=sim2,
                        )
                        try:
                            lon_select = df1["ref_lon"].values
                            lat_select = df1["ref_lat"].values
                        except (KeyError, ValueError) as e:
                            logging.debug(f"Using sim coordinates instead of ref coordinates: {e}")
                            lon_select = df1["sim_lon"].values
                            lat_select = df1["sim_lat"].values
                        diff_df = pd.DataFrame(
                            {
                                "ID": df1["ID"],
                                "lat": lat_select,
                                "lon": lon_select,
                                f"{score}_diff": diff,
                            }
                        )

                        output_file = os.path.join(
                            dir_path,
                            diff_station_difference_filename(
                                evaluation_item,
                                ref_source,
                                sim1,
                                sim_varname_1,
                                sim2,
                                sim_varname_2,
                                score,
                            ),
                        )
                        diff_df = Convert_Type.convert_Frame(diff_df)
                        _write_csv_atomic(diff_df, output_file, index=False)

                    except Exception as e:
                        logging.error(f"Error processing station score {score} for {sim1} vs {sim2}: {e}")
                        raise
