#!/usr/bin/env python3

from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)
import multiprocessing
import datetime
import os
import numpy as np
import pandas as pd

from main import ALL_BANDS, RADIOMETRIC_BANDS

class DataProcessor:

    def __init__(self, data, logger):
        self.data = data
        self.logger = logger
        pass
        
    def parse(pixels, columns_types):
        """
        Parse raw pixel data from Earth Engine into a DataFrame.

        Args:
            pixels: Raw pixel data from Earth Engine
            columns_types: Dictionary mapping column names to data types

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
        dataframe = dataframe[["id", "longitude", "latitude", "time"] + ALL_BANDS]
        dataframe[ALL_BANDS] = dataframe[ALL_BANDS].astype(float)
        dataframe.reset_index(drop=True, inplace=True)
        dataframe.fillna(-1, inplace=True)

        # Convert columns to specified types
        for column, dtype in columns_types.items():
            if column in dataframe.columns:
                dataframe[column] = dataframe[column].astype(dtype)

        return dataframe.reset_index(drop=True)


    def get_time_windows(start_date, end_date, steps):
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

    def process_dataframe(self, df, dates, parcel_id):
        """
        Process the downloaded dataframe into the final array format.

        Args:
            df: DataFrame with raw satellite data
            dates: DatetimeIndex of dates to include
            parcel_id: Parcel ID
            logger: Logger object

        Returns:
            numpy.ndarray: Processed array or None if processing failed
        """
        if len(df) == 0:
            self.logger.error(f"Empty dataframe received for parcel {parcel_id}")
            return None

        # Calculate number of unique points
        n_points = df.groupby(["longitude", "latitude"]).ngroup().nunique()
        if n_points < 100:
            self.logger.error(
                f"Insufficient unique points ({n_points}/100) for parcel {parcel_id}"
            )
            return None

        # Prepare time series data
        df["ID_TS"] = df.groupby(["longitude", "latitude"]).ngroup()
        df["doa"] = pd.to_datetime(df["time"], unit="ms").dt.date
        df = df.set_index(["ID_TS", "doa"]).sort_index()

        # Sample 100 time series
        ids = df.index.get_level_values(0).unique()
        ids = np.random.choice(ids, size=100, replace=False)
        df = df.loc[ids]

        # Keep only needed bands and add NDVI
        df = df[RADIOMETRIC_BANDS + ["MSK_CLDPRB", "SCL"]]
        df["NDVI"] = (df["B8"] - df["B4"]) / (df["B8"] + df["B4"])

        # Clean data: remove invalid values, cloudy pixels, and poor quality data
        df = df[~df.isna().any(axis=1)]
        df = df[~df.isin([-1]).any(axis=1)]
        df = df[df["MSK_CLDPRB"] == 0]
        df = df[~df["SCL"].isin([3, 7, 8, 9, 10, 11])]  # Filter out low quality pixels
        df = df.drop(columns=["MSK_CLDPRB", "SCL"])
        df = df[~df.index.duplicated(keep="first")]
        df = df.reset_index(level=1)

        # Convert types and interpolate missing values
        df[RADIOMETRIC_BANDS + ["NDVI"]] = df[RADIOMETRIC_BANDS + ["NDVI"]].astype(
            "float64"
        )
        df = df.groupby(level=0, group_keys=True)[
            RADIOMETRIC_BANDS + ["NDVI"] + ["doa"]
        ].apply(
            lambda x: (
                x.set_index("doa")
                .reindex(dates)
                .interpolate(method="linear", limit_direction="both")
                .iloc[::5, :]  # Sample every 5th date
            )
        )

        # Filter to desired date range
        df = df[
            (df.index.get_level_values(1) >= "2022-02-01")
            & (df.index.get_level_values(1) <= "2022-11-30")
        ]

        # Add ID_RPG and convert types
        df["ID_RPG"] = parcel_id
        df = df.reset_index()
        df["ID_TS"] = df["ID_TS"].astype("int16")
        df["ID_RPG"] = df["ID_RPG"].astype("int32")
        df[RADIOMETRIC_BANDS] = df[RADIOMETRIC_BANDS].astype("int16")
        df["NDVI"] = df["NDVI"].astype("float64")

        # Verify final shape
        expected_rows = 6000  # 100 timeseries * 60 dates
        if len(df) != expected_rows:
            self.logger.error(
                f"Final shape mismatch for parcel {parcel_id}. Expected {expected_rows}, got {len(df)}"
            )
            return None

        # Return as reshaped numpy array: 100 points x 60 dates x 12 bands
        return df[RADIOMETRIC_BANDS].to_numpy().reshape(100, 60, 12)

    def download_and_process_worker(self, args, config, outfolder, dates, ee_client):
        """
        Worker function for parallel processing of parcels.

        Args:
            args: Tuple of (index, row) from DataFrame.iterrows()
            config: Configuration dictionary
            outfolder: Output folder path
            dates: DatetimeIndex of dates to include
            logger: Logger object

        Returns:
            bool: True if processing was successful, False otherwise
        """
        index, row = args
        try:
            outfile = outfolder / f"{int(index) // 5000}/{index}.npy"
            if outfile.exists():
                self.logger.debug(
                    f"File for parcel {row['ID_PARCEL']} already exists. Skipping..."
                )
                return True

            # Download data
            region = ee_client.shapely2ee(row["geometry"])
            raw_df = ee_client.retrieve_data(region, row, config)

            # Process data
            processed_array = self.process_dataframe(raw_df, dates, int(index))
            if processed_array is None:
                self.logger.error(f"Processing failed for parcel {row['ID_PARCEL']}")
                return False
            else:
                os.makedirs(os.path.dirname(outfile), exist_ok=True)
                np.save(outfile, processed_array)
                return True

        except Exception as e:
            self.logger.error(f"Error processing parcel {index}: {e}")
            return False

    def worker_wrapper(self, args, config, outfolder, dates, ee_client):
        """
        Wrapper around download_and_process_worker that unpacks arguments for multiprocessing.

        Args:
            args: Tuple of (index, row) from DataFrame.iterrows()
            config: Configuration dictionary
            outfolder: Output folder path
            dates: DatetimeIndex of dates to include
            logger: Logger object

        Returns:
            bool: True if processing was successful, False otherwise
        """
        return self.download_and_process_worker(args, config, outfolder, dates, ee_client)

    def process_parcels(self, df, config, outfolder, dates, n_workers=60, ee_client=None):
        """
        Process parcels in parallel.

        Args:
            df: GeoDataFrame with parcels
            config: Configuration dictionary
            outfolder: Output folder path
            dates: DatetimeIndex of dates
            logger: Logger object
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

        self.logger.info(f"Starting processing of {len(df)} parcels with {n_workers} workers")

        # Use functools.partial to create a function with fixed arguments except the first one
        import functools

        worker_func = functools.partial(
            self.worker_wrapper, config=config, outfolder=outfolder, dates=dates, logger=self.logger, ee_client=ee_client
        )

        try:
            with progress:
                task = progress.add_task("[cyan]Processing parcels...", total=len(df))
                for result in pool.imap_unordered(worker_func, df.iterrows()):
                    progress.update(task, advance=1)
        except AttributeError as e:
            if "'NoneType' object has no attribute 'dumps'" in str(e):
                # This error occurs during Pool cleanup and can be safely ignored
                self.logger.info("Ignoring Pool cleanup AttributeError")
            else:
                raise
        finally:
            pool.close()
            pool.join()

    def filter_by_area(df, area_min, area_max):
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
