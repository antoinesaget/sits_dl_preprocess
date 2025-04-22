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
import argparse
import mmap_ninja
import numpy as np

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
DEFAULT_DIRECTORY_PATHS = config["paths"]


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

    logger.info("Initializing classes")
    # Initialize EarthEngineClient, DataProcessor, and FileManager classes
    ee_client = EarthEngineClient()
    ee_client.initialize_earth_engine(logger)
    processor = DataProcessor()
    file_manager = FileManager()

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Satellite imagery processing pipeline")
    parser.add_argument("--current_dir", default=DEFAULT_DIRECTORY_PATHS["current_dir"], help="Base directory")
    parser.add_argument("--sample_parquet", default=DEFAULT_DIRECTORY_PATHS["sample_parquet"], help="Path to sample parquet file")
    parser.add_argument("--processed_arrays_folder", default=DEFAULT_DIRECTORY_PATHS["processed_arrays_folder"], help="Folder for processed arrays")
    parser.add_argument("--memmap_folder", default=DEFAULT_DIRECTORY_PATHS["memmap_folder"], help="Folder for memory-mapped arrays")
    parser.add_argument("--filtered_folder", default=DEFAULT_DIRECTORY_PATHS["filtered_folder"], help="Folder for filtered shapefiles")
    parser.add_argument("--filtered_shp_path", default=DEFAULT_DIRECTORY_PATHS["filtered_shp_path"], help="Path to filtered shapefile")
    args = parser.parse_args()

    # Define paths - using sample parquet file
    current_dir = Path(args.current_dir)
    sample_parquet = current_dir / args.sample_parquet
    processed_arrays_folder = current_dir / args.processed_arrays_folder
    memmap_folder = current_dir / args.memmap_folder
    filtered_folder = current_dir / args.filtered_folder
    filtered_shp_path = current_dir / args.filtered_shp_path

    # Create necessary directories
    os.makedirs(processed_arrays_folder, exist_ok=True)
    os.makedirs(memmap_folder, exist_ok=True)
    os.makedirs(filtered_folder, exist_ok=True)

    # Generate date range for the full year
    dates = pd.date_range(DEFAULT_CONFIG["start"], DEFAULT_CONFIG["end"], freq="D", name="doa")

    # Load parcels from sample parquet file
    logger.info("Loading parcel data from sample file")
    df = gpd.read_parquet(sample_parquet).reset_index()


    # Filter by area
    logger.info("Filtering parcels by area")
    df = processor.filter_by_area(df, DEFAULT_CONFIG["area_min"], DEFAULT_CONFIG["area_max"])
    print(df.head())

    # Save filtered shapefile
    logger.info(f"Saving filtered parcels to {filtered_shp_path}")
    df.to_file(filtered_shp_path, driver="ESRI Shapefile")

    # Convert to geographic coordinates for Earth Engine
    df = df.to_crs("epsg:4326")

    # Process parcels
    processor.process_parcels(df, DEFAULT_CONFIG, processed_arrays_folder, dates, logger, ee_client, RADIOMETRIC_BANDS, ALL_BANDS)

    # Filter and save valid parcels
    logger.info(f"Filtering and saving valid parcels to {processed_arrays_folder}")
    df = file_manager.filter_and_save_valid_parcels(df, processed_arrays_folder, current_dir, logger)

    # Create memory-mapped array
    logger.info(f"Creating memory-mapped array into {memmap_folder}")
    memmap = file_manager.create_memmap(df, processed_arrays_folder, memmap_folder, logger)

    print("Sample data from memory-mapped array:")
    print(memmap[:2, :5, :3])

    logger.info("Processing pipeline completed successfully")

if __name__ == "__main__":
    main()
