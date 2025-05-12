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

import geopandas as gpd
import hydra
import pandas as pd

from data_processing import DataProcessor
from earth_engine import EarthEngineClient
from file_operations import FileManager


class InfoOnlyFilter(logging.Filter):
    def filter(self, record):
        # Allow only INFO-level messages
        return record.levelno == logging.INFO or record.levelno == logging.ERROR


def setup_logging(log_file: str = "download_process.log") -> logging.Logger:
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
    stream_handler.addFilter(InfoOnlyFilter())
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.WARNING)
    logger.addHandler(file_handler)

    # Remove previous basic config
    for handler in logging.getLogger().handlers[:]:
        logging.getLogger().removeHandler(handler)

    return logger


@hydra.main(version_base=None, config_path="", config_name="config")
def main(data: dict):
    """
    Main function to execute the download and processing pipeline.
    Args:
        data: Configuration data loaded with Hydra from config.yaml
    """

    # Setup logging
    logger = setup_logging()

    # Define paths - using sample parquet file
    current_dir = Path(data.paths.current_dir)
    polygons_processed_folder = current_dir / data.paths.polygons_processed_folder
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

    if not data.default.ee_project_name:
        logger.error(
            "earth engine project name is required (ee-project-name=your_project_name)"
        )
        return

    # Initialize EarthEngineClient, DataProcessor, and FileManager classes
    logger.info("Initializing classes")
    ee_client = EarthEngineClient(
        logger, data.default, list(data.bands.radiometric_bands + data.bands.misc_bands)
    )
    ee_client.initialize_earth_engine(data.default.ee_project_name)
    processor = DataProcessor(
        logger,
        data.default,
        list(data.bands.radiometric_bands),
        list(data.bands.radiometric_bands + data.bands.misc_bands),
        processed_arrays_folder,
        dates,
        ee_client,
    )
    file_manager = FileManager(logger, processed_arrays_folder)

    if data.reset_folders:
        file_manager.reset_folders(data, logger)

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
    processor.process_parcels(df)

    # Filter and save valid parcels
    logger.info(f"Filtering and saving valid parcels to {processed_arrays_folder}")
    df = file_manager.filter_and_save_valid_parcels(df, polygons_processed_folder)

    # Create memory-mapped array
    logger.info(f"Creating memory-mapped array into {memmap_folder}")
    memmap = file_manager.create_memmap(df, memmap_folder)

    logger.info("Processing pipeline completed successfully")


if __name__ == "__main__":
    main()
