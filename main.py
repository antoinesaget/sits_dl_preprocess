#!/usr/bin/env python3
"""
Satellite imagery download and processing script for FranceCrops dataset.

This script downloads Sentinel-2 satellite imagery from Google Earth Engine,
processes it into a standardized format, and saves it as memory-mapped arrays
for efficient access.
"""

import logging
import os
import geopandas as gpd
import pandas as pd
from pathlib import Path
import yaml

from data_processing import DataProcessor
from earth_engine import EarthEngineClient
from file_operations import FileManager

with open("config.yaml", "r") as config_file:
    config = yaml.safe_load(config_file)

# Band definitions
RADIOMETRIC_BANDS = config["bands"]["radiometric_bands"]
MISC_BANDS = config["bands"]["misc_bands"]
ALL_BANDS = RADIOMETRIC_BANDS + MISC_BANDS

# Default configuration
DEFAULT_CONFIG = config["default"]

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

def main():
    """Main function to execute the download and processing pipeline."""
    # Setup logging
    logger = setup_logging()

    # Initialize Earth Engine
    ee_client = EarthEngineClient(logger)

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
    dates = pd.date_range(DEFAULT_CONFIG["start"], DEFAULT_CONFIG["end"], freq="D", name="doa")

    # Load parcels from sample parquet file
    logger.info("Loading parcel data from sample file")
    df = gpd.read_parquet(sample_parquet).reset_index()

    processor = DataProcessor(logger)
    # Filter by area
    logger.info("Filtering parcels by area")
    df = processor.filter_by_area(df, DEFAULT_CONFIG["area_min"], DEFAULT_CONFIG["area_max"])
    print(df.head())

    # Save filtered shapefile
    filtered_shp_path = filtered_folder / "polygons_filtered.shp"
    logger.info(f"Saving filtered parcels to {filtered_shp_path}")
    df.to_file(filtered_shp_path, driver="ESRI Shapefile")

    # Convert to geographic coordinates for Earth Engine
    df = df.to_crs("epsg:4326")

    # Process parcels
    processor.process_parcels(df, DEFAULT_CONFIG, processed_arrays_folder, dates, ee_client)

    file_manager = FileManager(logger)
    # Filter and save valid parcels
    df = file_manager.filter_and_save_valid_parcels(df, processed_arrays_folder, current_dir)

    # Create memory-mapped array
    file_manager.create_memmap(df, processed_arrays_folder, memmap_folder)

    logger.info("Processing pipeline completed successfully")

if __name__ == "__main__":
    main()
