#!/usr/bin/env python3

import datetime
import multiprocessing
import os
from logging import Logger

import geopandas as gpd
import numpy as np
import pandas as pd
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)


class DataProcessor:
    def __init__(
        self,
        logger: Logger,
        data: dict,
        radiometric_bands: list,
        all_bands: list,
        outfolder: str,
        dates: pd.DatetimeIndex,
        ee_client,
    ):
        """
        Initialize the DataProcessor with default configuration.
        This class is responsible for processing satellite data from Earth Engine.
        Args:
            logger: Logger object for recording status
            data: Configuration data
            radiometric_bands: List of radiometric bands
            all_bands: List of all bands available in the dataset
            outfolder: Output folder path for saving processed data
            dates: DatetimeIndex of dates to include in the processing
            ee_client: EarthEngineClient object for interacting with Google Earth Engine
        """
        self.logger = logger
        self.data = data
        self.radiometric_bands = radiometric_bands
        self.all_bands = all_bands
        self.outfolder = outfolder
        self.dates = dates
        self.ee_client = ee_client

    def parse(self, pixels: list[list]) -> pd.DataFrame:
        """
        Parse raw pixel data from Earth Engine into a DataFrame.

        Args:
            pixels: Raw pixel data from Earth Engine

        Returns:
            pd.DataFrame: Processed DataFrame
        """
        dataframe = pd.DataFrame(pixels[1:], columns=pixels[0])
        dataframe["TILE"] = dataframe["id"].str.split("_").str[2]

        # Filter to one tile if multiple exist
        if dataframe["TILE"].nunique() > 1:
            tile = np.sort(dataframe["TILE"].unique())[0]
            dataframe = dataframe[dataframe["TILE"] == tile]

        # Select and process columns
        dataframe = dataframe[
            ["id", "longitude", "latitude", "time", "CLOUD_PROB"] + self.all_bands
        ]
        dataframe[self.all_bands] = dataframe[self.all_bands].astype(float)
        dataframe.reset_index(drop=True, inplace=True)
        dataframe.fillna(-1, inplace=True)

        # Convert columns to specified types
        for column, dtype in self.data.columns_types.items():
            if column in dataframe.columns:
                dataframe[column] = dataframe[column].astype(dtype)
        if "CLOUD_PROB" in dataframe.columns:
            dataframe["CLOUD_PROB"] = dataframe["CLOUD_PROB"].astype("int8")
        return dataframe.reset_index(drop=True)

    def get_time_windows(self, start_date: str, end_date: str, steps: int) -> tuple:
        """
        Split a time range into smaller windows.

        Args:
            start_date: Start date string (YYYY-MM-DD)
            end_date: End date string (YYYY-MM-DD)
            steps: Number of days in each window

        Returns:
            Tuple of lists: (start_dates, end_dates)
        """
        starts = pd.date_range(start_date, end_date, freq=f"{steps}D")
        ends = starts + datetime.timedelta(days=-1)
        ends = ends[1:]

        starts = starts.strftime("%Y-%m-%d").tolist()
        ends = ends.strftime("%Y-%m-%d").tolist()
        ends.append(pd.to_datetime(end_date).strftime("%Y-%m-%d"))

        return starts, ends

    def process_dataframe(
        self, df: pd.DataFrame, parcel_id: int, pracel_index: int
    ) -> np.ndarray:
        """
        Process the downloaded dataframe into the final array format.

        Args:
            df: DataFrame with raw satellite data
            parcel_id: Parcel ID

        Returns:
            numpy.ndarray: Processed array or None if processing failed
        """
        if len(df) == 0:
            self.logger.warning(f"Empty dataframe received for parcel {parcel_id}")
            return None

        # Calculate number of unique points
        n_points = df.groupby(["longitude", "latitude"]).ngroup().nunique()
        if n_points < self.data.points:
            self.logger.warning(
                f"Insufficient unique points ({n_points}/{self.data.points}) for parcel {parcel_id}"
            )
            return None

        # Prepare time series data
        df["ID_TS"] = df.groupby(["longitude", "latitude"]).ngroup()
        df["doa"] = pd.to_datetime(df["time"], unit="ms").dt.date
        df = df.set_index(["ID_TS", "doa"]).sort_index()

        # Sample time series
        ids = df.index.get_level_values(0).unique()
        ids = np.random.choice(ids, size=self.data.points, replace=False)
        df = df.loc[ids]

        # Keep only needed bands and add NDVI
        df = df[self.radiometric_bands + ["MSK_CLDPRB", "CLOUD_PROB", "SCL"]]
        df["NDVI"] = (df["B8"] - df["B4"]) / (df["B8"] + df["B4"])

        # Clean data: remove invalid values, cloudy pixels, and poor quality data
        df = df[~df.isna().any(axis=1)]
        df = df[~df.isin([-1]).any(axis=1)]
        df = df[df["MSK_CLDPRB"] == 0]
        df = df[df["CLOUD_PROB"] < self.data.clouds_threshold]
        df = df[~df["SCL"].isin([3, 7, 8, 9, 10, 11])]  # Filter out low quality pixels
        df = df.drop(columns=["MSK_CLDPRB", "CLOUD_PROB", "SCL"])
        df = df[~df.index.duplicated(keep="first")]
        df = df.reset_index(level=1)

        if df.empty:
            self.logger.warning(
                f"Empty dataframe after filtering for parcel {parcel_id}"
            )
            return None

        # Convert types and interpolate missing values
        df[self.radiometric_bands + ["NDVI"]] = df[
            self.radiometric_bands + ["NDVI"]
        ].astype("float64")
        df = df.groupby(level=0, group_keys=True)[
            self.radiometric_bands + ["NDVI"] + ["doa"]
        ].apply(
            lambda x: (
                x.set_index("doa")
                .reindex(self.dates)
                .interpolate(method="linear", limit_direction="both")
                .iloc[:: self.data.days_interval, :]  # Sample every 5th date
            )
        )
        # Filter to desired date range
        if df.index.nlevels == 2:
            df = df[
                (df.index.get_level_values(1) >= self.data.filter_start)
                & (df.index.get_level_values(1) <= self.data.filter_end)
            ]
        else:
            df = df.reset_index()
            df = df[
                (df["doa"] >= self.data.filter_start)
                & (df["doa"] <= self.data.filter_end)
            ]

        # Add ID_RPG and convert types
        df["ID_RPG"] = pracel_index
        df = df.reset_index()
        df["ID_TS"] = df["ID_TS"].astype("int16")
        df["ID_RPG"] = df["ID_RPG"].astype("int32")
        df[self.radiometric_bands] = df[self.radiometric_bands].astype("int16")
        df["NDVI"] = df["NDVI"].astype("float64")

        # Verify final shape
        filter_start = datetime.datetime.strptime(
            self.data.filter_start, "%Y-%m-%d"
        ).date()
        filter_end = datetime.datetime.strptime(self.data.filter_end, "%Y-%m-%d").date()
        diff = (filter_end - filter_start).days
        nb_dates = round(diff / self.data.days_interval)
        expected_rows = nb_dates * self.data.points
        if len(df) != expected_rows:
            self.logger.warning(
                f"Final shape mismatch for parcel {parcel_id}. Expected {expected_rows}, got {len(df)}"
            )
            return None

        # Return as reshaped numpy array: nb of points x nb of dates x nb of bands (default: 100 x 60 x 12)
        return (
            df[self.radiometric_bands]
            .to_numpy()
            .reshape(self.data.points, nb_dates, len(self.radiometric_bands))
        )

    def download_and_process_worker(self, args: tuple) -> bool:
        """
        Worker function for parallel processing of parcels.

        Args:
            args: Tuple of (index, row) from DataFrame.iterrows()

        Returns:
            bool: True if processing was successful, False otherwise
        """
        index, row = args
        try:
            outfile = self.outfolder / f"{int(index) // 5000}/{index}.npy"
            if outfile.exists():
                self.logger.debug(
                    f"File for parcel {row['ID_PARCEL']} already exists. Skipping..."
                )
                return True

            # Download data
            region = self.ee_client.shapely2ee(row["geometry"])
            raw_df = self.ee_client.retrieve_data(region, row, self)

            # Process data
            processed_array = self.process_dataframe(
                raw_df, row["ID_PARCEL"], int(index)
            )
            if processed_array is None:
                self.logger.warning(f"Processing failed for parcel {row['ID_PARCEL']}")
                return False
            else:
                os.makedirs(os.path.dirname(outfile), exist_ok=True)
                np.save(outfile, processed_array)
                return True

        except Exception as e:
            self.logger.warning(f"Error processing parcel {row['ID_PARCEL']}: {e}")
            return False

    def worker_wrapper(self, args: tuple) -> bool:
        """
        Wrapper around download_and_process_worker that unpacks arguments for multiprocessing.

        Args:
            args: Tuple of (index, row) from DataFrame.iterrows()

        Returns:
            bool: True if processing was successful, False otherwise
        """
        return self.download_and_process_worker(args)

    def process_parcels(self, df: gpd.GeoDataFrame, n_workers: int = 60) -> None:
        """
        Process parcels in parallel.

        Args:
            df: GeoDataFrame with parcels
            n_workers: Number of parallel workers

        Returns:
            None
        """
        pool = multiprocessing.Pool(n_workers)

        # Setup progress display
        progress = Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("({task.remaining} remaining)"),
            TaskProgressColumn(),
            TimeRemainingColumn(),
        )

        self.logger.info(
            f"Starting processing of {len(df)} parcels with {n_workers} workers"
        )

        # Use functools.partial to create a function with fixed arguments except the first one
        import functools

        worker_func = functools.partial(self.worker_wrapper)

        try:
            with progress:
                task = progress.add_task("[cyan]Processing parcels...", total=len(df))
                for result in pool.imap_unordered(worker_func, df.iterrows()):
                    progress.update(task, advance=1)
        except AttributeError as e:
            if "'NoneType' object has no attribute 'dumps'" in str(e):
                # This error occurs during Pool cleanup and can be safely ignored
                self.logger.warning("Ignoring Pool cleanup AttributeError")
            else:
                raise
        finally:
            self.logger.info(f"Processing completed for {len(df)} parcels")
            pool.close()
            pool.join()

    def filter_by_area(
        self, df: gpd.GeoDataFrame, area_min: int, area_max: int
    ) -> gpd.GeoDataFrame:
        """
        Filter parcels by area in hectares.

        Args:
            df: GeoDataFrame with parcels
            area_min: Minimum area in hectares
            area_max: Maximum area in hectares

        Returns:
            GeoDataFrame: Filtered parcels
        """
        # Convert to metric CRS for area calculation
        original_crs = df.crs
        df = df.to_crs("epsg:2154")
        df["AREA_HA"] = df.geometry.area / 10000

        # Filter by area
        df = df[(df["AREA_HA"] >= area_min) & (df["AREA_HA"] <= area_max)]

        # Return to original CRS
        return df.to_crs(original_crs)
