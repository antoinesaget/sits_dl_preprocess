#!/usr/bin/env python3
"""
Satellite imagery download and processing script for FranceCrops dataset.

This script downloads Sentinel-2 satellite imagery from Google Earth Engine,
processes it into a standardized format, and saves it as memory-mapped arrays
for efficient access.
"""

import datetime
import logging
import multiprocessing
import os
import time
from pathlib import Path

import ee
import geopandas as gpd
import mmap_ninja
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

# Band definitions
RADIOMETRIC_BANDS = [
    "B1",
    "B2",
    "B3",
    "B4",
    "B5",
    "B6",
    "B7",
    "B8",
    "B8A",
    "B9",
    "B11",
    "B12",
]
MISC_BANDS = [
    "AOT",
    "WVP",
    "SCL",
    "TCI_R",
    "TCI_G",
    "TCI_B",
    "MSK_CLDPRB",
    "MSK_SNWPRB",
    "QA60",
]
ALL_BANDS = RADIOMETRIC_BANDS + MISC_BANDS

# Default configuration
DEFAULT_CONFIG = {
    "start": "2022-01-01",
    "end": "2022-12-31",
    "collection": "COPERNICUS/S2_SR_HARMONIZED",
    "scale": 10,
    "columns_types": {
        "SCL": "category",
        "MSK_CLDPRB": "int8",
        "B1": "int16",
        "B2": "int16",
        "B3": "int16",
        "B4": "int16",
        "B5": "int16",
        "B6": "int16",
        "B7": "int16",
        "B8": "int16",
        "B8A": "int16",
        "B9": "int16",
        "B11": "int16",
        "B12": "int16",
    },
    "steps": 64,
    "area_min": 0.1,  # Minimum parcel area in hectares
    "area_max": 40,  # Maximum parcel area in hectares
}


def setup_logging(log_file="download_process.log"):
    """
    Configure logging for the application.

    Args:
        log_file: Path to the log file

    Returns:
        logger: Configured logger object
    """
    # Set root logger to ERROR level for all external libraries
    logging.getLogger().setLevel(logging.ERROR)

    # Configure module's logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # Add handlers
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Remove previous basic config
    for handler in logging.getLogger().handlers[:]:
        logging.getLogger().removeHandler(handler)

    return logger


def initialize_earth_engine(logger):
    """
    Initialize Google Earth Engine with authentication if needed.

    Args:
        logger: Logger object for recording status
    """
    try:
        ee.Initialize(
            opt_url="https://earthengine-highvolume.googleapis.com",
            project="ee-magzoumov",
        )
        logger.info("Earth Engine initialized successfully")
    except Exception as e:
        logger.warning(
            f"Initial EE initialization failed, attempting authentication: {e}"
        )
        ee.Authenticate()
        ee.Initialize(
            opt_url="https://earthengine-highvolume.googleapis.com",
            project="ee-magzoumov",
        )
        logger.info("Earth Engine initialized after authentication")


def shapely2ee(geometry):
    """
    Convert Shapely geometry to Earth Engine geometry.

    Args:
        geometry: Shapely geometry object

    Returns:
        ee.Geometry: Earth Engine geometry object
    """
    pt_list = list(zip(*geometry.exterior.coords.xy))
    return ee.Geometry.Polygon(pt_list)


def query(region, start, end, collection, scale):
    """
    Query Earth Engine for satellite imagery.

    Args:
        region: Earth Engine geometry
        start: Start date (YYYY-MM-DD)
        end: End date (YYYY-MM-DD)
        collection: Earth Engine collection name
        scale: Pixel scale in meters

    Returns:
        List of image data
    """
    images = (
        ee.ImageCollection(collection)
        .filterDate(start, end)
        .filterBounds(region)
        .filter(ee.Filter.eq("GENERAL_QUALITY", "PASSED"))
        .select(ALL_BANDS)
    )

    sampled_points = ee.FeatureCollection.randomPoints(
        **{"region": region, "points": 200, "seed": 42}
    )

    return images.getRegion(sampled_points, scale).getInfo()


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


def retrieve_data(region, row, config, logger):
    """
    Retrieve satellite data from Earth Engine with automatic retry for large areas.

    Args:
        region: Earth Engine geometry
        row: DataFrame row containing parcel info
        config: Configuration dictionary
        logger: Logger object

    Returns:
        pd.DataFrame: Downloaded satellite data
    """
    retry = False
    steps = config["steps"]
    parcel_id = row["ID_PARCEL"]

    while True:
        if steps < 4:
            logger.error(f"Parcel {parcel_id} too large to process. Skipping...")
            break

        try:
            if retry:
                # Try with smaller time windows
                starts, ends = get_time_windows(config["start"], config["end"], steps)
                dataframe = pd.DataFrame()
                logger.debug(
                    f"Retrying with {len(starts)} time steps for parcel {parcel_id}"
                )

                for _start, _end in zip(starts, ends):
                    getinfo_dict = query(
                        region, _start, _end, config["collection"], config["scale"]
                    )
                    dataframe_local = parse(getinfo_dict, config["columns_types"])
                    if len(dataframe_local) > 0:
                        dataframe = pd.concat([dataframe, dataframe_local])

                return dataframe.reset_index(drop=True)
            else:
                # Try with one large time window
                getinfo_dict = query(
                    region,
                    config["start"],
                    config["end"],
                    config["collection"],
                    config["scale"],
                )
                dataframe = parse(getinfo_dict, config["columns_types"])
                return dataframe

        except ee.ee_exception.EEException as e:
            if "ImageCollection.getRegion: Too many values:" in str(e):
                if retry:
                    steps = steps // 2
                    logger.debug(f"Reducing steps to {steps} for parcel {parcel_id}")
                retry = True
                continue

            if "Too Many Requests" in str(e):
                logger.debug("Rate limit hit, waiting 1 second...")
                time.sleep(1)
                continue

            logger.error(f"Earth Engine error for parcel {parcel_id}: {e}")
            break

        except Exception as e:
            logger.error(f"Unexpected error processing parcel {parcel_id}: {e}")
            break

    return pd.DataFrame()


def process_dataframe(df, dates, parcel_id, logger):
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
        logger.error(f"Empty dataframe received for parcel {parcel_id}")
        return None

    # Calculate number of unique points
    n_points = df.groupby(["longitude", "latitude"]).ngroup().nunique()
    if n_points < 100:
        logger.error(
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
        logger.error(
            f"Final shape mismatch for parcel {parcel_id}. Expected {expected_rows}, got {len(df)}"
        )
        return None

    # Return as reshaped numpy array: 100 points x 60 dates x 12 bands
    return df[RADIOMETRIC_BANDS].to_numpy().reshape(100, 60, 12)


def download_and_process_worker(args, config, outfolder, dates, logger):
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
            logger.debug(
                f"File for parcel {row['ID_PARCEL']} already exists. Skipping..."
            )
            return True

        # Download data
        region = shapely2ee(row["geometry"])
        raw_df = retrieve_data(region, row, config, logger)

        # Process data
        processed_array = process_dataframe(raw_df, dates, int(index), logger)
        if processed_array is None:
            logger.error(f"Processing failed for parcel {row['ID_PARCEL']}")
            return False
        else:
            os.makedirs(os.path.dirname(outfile), exist_ok=True)
            np.save(outfile, processed_array)
            return True

    except Exception as e:
        logger.error(f"Error processing parcel {index}: {e}")
        return False


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


def worker_wrapper(args, config, outfolder, dates, logger):
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
    return download_and_process_worker(args, config, outfolder, dates, logger)


def process_parcels(df, config, outfolder, dates, logger, n_workers=60):
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

    logger.info(f"Starting processing of {len(df)} parcels with {n_workers} workers")

    # Use functools.partial to create a function with fixed arguments except the first one
    import functools

    worker_func = functools.partial(
        worker_wrapper, config=config, outfolder=outfolder, dates=dates, logger=logger
    )

    try:
        with progress:
            task = progress.add_task("[cyan]Processing parcels...", total=len(df))
            for result in pool.imap_unordered(worker_func, df.iterrows()):
                progress.update(task, advance=1)
    except AttributeError as e:
        if "'NoneType' object has no attribute 'dumps'" in str(e):
            # This error occurs during Pool cleanup and can be safely ignored
            logger.info("Ignoring Pool cleanup AttributeError")
        else:
            raise
    finally:
        pool.close()
        pool.join()


def filter_and_save_valid_parcels(df, input_folder, output_path, logger):
    """
    Filter parcels to keep only those with valid processed files.

    Args:
        df: GeoDataFrame with parcels
        input_folder: Folder containing processed .npy files
        output_path: Path for dataset files
        logger: Logger object

    Returns:
        GeoDataFrame: Filtered parcels
    """
    logger.info("Starting validation")

    valid_indices = []
    for i in range(len(df)):
        idx = df.iloc[i]["ID_PARCEL"]
        if (input_folder / f"{idx // 5000}/{idx}.npy").exists():
            valid_indices.append(i)

    df = df.iloc[valid_indices].reset_index(drop=True)
    logger.info(f"Filtered to {len(df)} rows with existing .npy files")

    # Save filtered dataset
    shapefile_path = output_path / "polygons_processed.shp"
    parquet_path = output_path / "polygons_processed.parquet"

    logger.info(f"Saving processed parcels to {shapefile_path} and {parquet_path}")

    # Ensure output directory exists
    os.makedirs(output_path, exist_ok=True)

    # Save files
    df.to_file(shapefile_path, driver="ESRI Shapefile")
    df.to_parquet(parquet_path)

    return df


def create_memmap(df, input_folder, output_folder, logger):
    """
    Convert individual .npy files to a memory-mapped array.

    Args:
        df: GeoDataFrame with valid parcels
        input_folder: Folder containing .npy files
        output_folder: Folder to save memory-mapped files
        logger: Logger object

    Returns:
        mmap_ninja.NpMemmap: Memory-mapped array
    """
    logger.info("Starting memmap conversion")

    # Ensure output directory exists
    os.makedirs(output_folder, exist_ok=True)

    # Function to get array by index
    def get_array(i):
        idx = df.iloc[i]["ID_PARCEL"]
        return np.load(input_folder / f"{idx // 5000}/{idx}.npy")

    # Create memory-mapped array from generator
    from tqdm.autonotebook import tqdm

    logger.info(f"Creating memory-mapped array from {len(df)} parcels")
    mmap_ninja.np_from_generator(
        out_dir=output_folder,
        sample_generator=map(get_array, tqdm(range(len(df)))),
        batch_size=16384,
    )

    # Open and verify the created memory-mapped array
    memmap = mmap_ninja.np_open_existing(output_folder)
    logger.info(f"Memmap conversion completed. Final shape: {memmap.shape}")

    return memmap


def main():
    """Main function to execute the download and processing pipeline."""
    # Setup logging
    logger = setup_logging()

    # Initialize Earth Engine
    initialize_earth_engine(logger)

    # Define paths - using sample parquet file
    current_dir = Path(".")
    sample_parquet = current_dir / "sample_agricultural_parcels_10k.parquet"
    processed_arrays_folder = current_dir / "processed_arrays"
    memmap_folder = current_dir / "out_memmap"
    filtered_folder = current_dir / "filtered"

    # Create necessary directories
    os.makedirs(processed_arrays_folder, exist_ok=True)
    os.makedirs(memmap_folder, exist_ok=True)
    os.makedirs(filtered_folder, exist_ok=True)
    # Generate date range for the full year
    dates = pd.date_range("2022-01-01", "2022-12-31", freq="D", name="doa")

    # Load parcels from sample parquet file
    logger.info("Loading parcel data from sample file")
    df = gpd.read_parquet(sample_parquet).reset_index()

    # Filter by area
    logger.info("Filtering parcels by area")
    df = filter_by_area(df, DEFAULT_CONFIG["area_min"], DEFAULT_CONFIG["area_max"])
    print(df.head())

    # Save filtered shapefile
    filtered_shp_path = filtered_folder / "polygons_filtered.shp"
    logger.info(f"Saving filtered parcels to {filtered_shp_path}")
    df.to_file(filtered_shp_path, driver="ESRI Shapefile")

    # Convert to geographic coordinates for Earth Engine
    df = df.to_crs("epsg:4326")

    # Process parcels
    process_parcels(df, DEFAULT_CONFIG, processed_arrays_folder, dates, logger)

    # Filter and save valid parcels
    df = filter_and_save_valid_parcels(df, processed_arrays_folder, current_dir, logger)

    # Create memory-mapped array
    create_memmap(df, processed_arrays_folder, memmap_folder, logger)

    logger.info("Processing pipeline completed successfully")


if __name__ == "__main__":
    main()
