#!/usr/bin/env python3
"""
Satellite imagery download and processing script for FranceCrops dataset.

This script downloads Sentinel-2 satellite imagery from Google Earth Engine,
processes it into a standardized format, and saves it as memory-mapped arrays
for efficient access.
"""

import logging
import os
from pathlib import Path
import hydra

import geopandas as gpd
import pandas as pd
import yaml

from data_processing import DataProcessor
from earth_engine import EarthEngineClient
from file_operations import FileManager


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
    logging.basicConfig()
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


@hydra.main(version_base=None, config_path="", config_name="config")
def main(data):
    """Main function to execute the download and processing pipeline."""

    # Setup logging
    logger = setup_logging()

    logger.info("Initializing classes")
    # Initialize EarthEngineClient, DataProcessor, and FileManager classes
    ee_client = EarthEngineClient()
    ee_client.initialize_earth_engine(logger, data.default.ee_project_name)
    processor = DataProcessor(data.default)
    file_manager = FileManager()

    # Define paths - using sample parquet file
    current_dir = Path(data.paths.current_dir)
    sample_parquet = current_dir / data.paths.sample_parquet
    processed_arrays_folder = current_dir / data.paths.processed_arrays_folder
    memmap_folder = current_dir / data.paths.memmap_folder
    filtered_folder = current_dir / data.paths.filtered_folder
    filtered_shp_path = current_dir / data.paths.filtered_shp_path

    # Create necessary directories
    os.makedirs(processed_arrays_folder, exist_ok=True)
    os.makedirs(memmap_folder, exist_ok=True)
    os.makedirs(filtered_folder, exist_ok=True)

    # Generate date range for the full year
    dates = pd.date_range(data.default.start, data.default.end, freq="D", name="doa")

    # Load parcels from sample parquet file
    logger.info("Loading parcel data from sample file")
    df = gpd.read_parquet(sample_parquet).reset_index()

    # Filter by area
    logger.info("Filtering parcels by area")
    df = processor.filter_by_area(df, data.default.area_min, data.default.area_max)
    print(df.head())

    # Save filtered shapefile
    logger.info(f"Saving filtered parcels to {filtered_shp_path}")
    df.to_file(filtered_shp_path, driver="ESRI Shapefile")

    # Convert to geographic coordinates for Earth Engine
    df = df.to_crs("epsg:4326")

    # Process parcels
    processor.process_parcels(
        df,
        data.default,
        processed_arrays_folder,
        dates,
        logger,
        ee_client,
        list(data.bands.radiometric_bands),
        list(data.bands.radiometric_bands + data.bands.misc_bands),
    )

    # Filter and save valid parcels
    logger.info(f"Filtering and saving valid parcels to {processed_arrays_folder}")
    df = file_manager.filter_and_save_valid_parcels(
        df, processed_arrays_folder, current_dir, logger
    )

    # Create memory-mapped array
    logger.info(f"Creating memory-mapped array into {memmap_folder}")
    memmap = file_manager.create_memmap(
        df, processed_arrays_folder, memmap_folder, logger
    )

    print("Sample data from memory-mapped array:")
    print(memmap[:2, :5, :3])

    logger.info("Processing pipeline completed successfully")


if __name__ == "__main__":
    main()
