#!/usr/bin/env python3
"""
Satellite imagery download and processing script for FranceCrops dataset.

This script downloads Sentinel-2 satellite imagery from Google Earth Engine,
processes it into a standardized format, and saves it as memory-mapped arrays
for efficient access.
"""

import argparse
import logging
import os
from datetime import datetime
from pathlib import Path

import geopandas as gpd
import pandas as pd
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


def validate_data():
    # Parse dates into datetime objects
    start = datetime.strptime(DEFAULT_CONFIG["start"], "%Y-%m-%d").date()
    filter_start = datetime.strptime(DEFAULT_CONFIG["filter_start"], "%Y-%m-%d").date()
    end = datetime.strptime(DEFAULT_CONFIG["end"], "%Y-%m-%d").date()
    filter_end = datetime.strptime(DEFAULT_CONFIG["filter_end"], "%Y-%m-%d").date()
    area_min = DEFAULT_CONFIG["area_min"]
    area_max = DEFAULT_CONFIG["area_max"]

    # Validate date ranges
    if start > end:
        raise ValueError("Start date must be earlier than end date.")
    if filter_start > filter_end:
        raise ValueError("Filter start date must be earlier than filter end date.")
    if start > filter_start:
        raise ValueError(
            "Start date must be earlier or equal to the filter_start date."
        )
    if end < filter_end:
        raise ValueError("End date must be later or equal to the filter_end date.")
    if (area_min < 0) or (area_max < 0):
        raise ValueError("Area min and max must be non-negative.")
    if area_min > area_max:
        raise ValueError("Area min must be less than or equal to area max.")


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Satellite imagery processing pipeline"
    )
    parser.add_argument(
        "--current_dir",
        default=DEFAULT_DIRECTORY_PATHS["current_dir"],
        help="Base directory",
    )
    parser.add_argument(
        "--sample_parquet",
        default=DEFAULT_DIRECTORY_PATHS["sample_parquet"],
        help="Path to sample parquet file",
    )
    parser.add_argument(
        "--processed_arrays_folder",
        default=DEFAULT_DIRECTORY_PATHS["processed_arrays_folder"],
        help="Folder for processed arrays",
    )
    parser.add_argument(
        "--memmap_folder",
        default=DEFAULT_DIRECTORY_PATHS["memmap_folder"],
        help="Folder for memory-mapped arrays",
    )
    parser.add_argument(
        "--filtered_folder",
        default=DEFAULT_DIRECTORY_PATHS["filtered_folder"],
        help="Folder for filtered shapefiles",
    )
    parser.add_argument(
        "--filtered_shp_path",
        default=DEFAULT_DIRECTORY_PATHS["filtered_shp_path"],
        help="Path to filtered shapefile",
    )
    return parser.parse_args()


def main():
    """Main function to execute the download and processing pipeline."""

    # Parse command-line arguments
    args = parse_arguments()

    # Validate configuration data
    validate_data()

    # Setup logging
    logger = setup_logging()

    logger.info("Initializing classes")
    # Initialize EarthEngineClient, DataProcessor, and FileManager classes
    ee_client = EarthEngineClient()
    ee_client.initialize_earth_engine(logger, DEFAULT_CONFIG["ee_project_name"])
    processor = DataProcessor()
    file_manager = FileManager()

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
    dates = pd.date_range(
        DEFAULT_CONFIG["start"], DEFAULT_CONFIG["end"], freq="D", name="doa"
    )

    # Load parcels from sample parquet file
    logger.info("Loading parcel data from sample file")
    df = gpd.read_parquet(sample_parquet).reset_index()

    # Filter by area
    logger.info("Filtering parcels by area")
    df = processor.filter_by_area(
        df, DEFAULT_CONFIG["area_min"], DEFAULT_CONFIG["area_max"]
    )
    print(df.head())

    # Save filtered shapefile
    logger.info(f"Saving filtered parcels to {filtered_shp_path}")
    df.to_file(filtered_shp_path, driver="ESRI Shapefile")

    # Convert to geographic coordinates for Earth Engine
    df = df.to_crs("epsg:4326")

    # Process parcels
    processor.process_parcels(
        df,
        DEFAULT_CONFIG,
        processed_arrays_folder,
        dates,
        logger,
        ee_client,
        RADIOMETRIC_BANDS,
        ALL_BANDS,
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
