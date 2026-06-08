import pandas as pd
import xarray as xr


class Convert_Type:
    def __init__(self):
        self.name = "DatasetPreprocessing"
        self.version = "0.1"
        self.release = "0.1"
        self.date = "Apr 2025"
        self.author = "Qingchen Xu / xuqingchen0@gmail.com"

    @staticmethod
    def convert_nc(ds: xr.DataArray | xr.Dataset) -> xr.DataArray | xr.Dataset:
        def _can_downcast_coord(name: str, coord: xr.DataArray) -> bool:
            # Time coordinates often arrive as numeric float64 before decode;
            # downcasting them to float32 loses sub-daily precision.
            return str(name).lower() != "time" and coord.dtype.kind == "f" and coord.dtype.itemsize == 8

        if isinstance(ds, xr.Dataset):
            ds = ds.copy(deep=False)
            for var in ds.data_vars:
                if ds[var].dtype.kind == "f" and ds[var].dtype.itemsize == 8:  # 检查是否 float64
                    ds[var] = ds[var].astype(dtype="float32", casting="same_kind", copy=False)
        else:
            if ds.dtype == "float64":
                ds = ds.astype(dtype="float32", casting="same_kind", copy=False)
        ds = ds.assign_coords(
            {
                coord: ds.coords[coord].astype("float32")
                for coord in ds.coords
                if _can_downcast_coord(coord, ds.coords[coord])
            }
        )
        return ds

    @staticmethod
    def convert_Frame(df: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
        if isinstance(df, pd.DataFrame):
            float64_cols = df.select_dtypes(include="float64").columns
            df[float64_cols] = df[float64_cols].astype("float32")
        else:
            if df.dtype == "float64":
                df = df.astype("float32")
        return df
