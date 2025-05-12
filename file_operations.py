#!/usr/bin/env python3

import os
from logging import Logger
from pathlib import Path

import geopandas as gpd
import mmap_ninja
import numpy as np


class FileManager:
    def __init__(self, logger: Logger, input_folder: str):
        """
        Initialize the FileManager class.
        This class is responsible for managing file operations related to parcel data.
        Args:
            logger: Logger object for recording status
            input_folder: Folder containing processed .npy files
        """
        self.logger = logger
        self.input_folder = input_folder

    def filter_and_save_valid_parcels(
        self, df: gpd.GeoDataFrame, output_path: str
    ) -> gpd.GeoDataFrame:
        """
        Filter parcels to keep only those with valid processed files.

        Args:
            df: GeoDataFrame with parcels
            output_path: Path for dataset files

        Returns:
            GeoDataFrame: Filtered parcels
        """
        self.logger.info("Starting validation")

        valid_indices = []
        for i in range(len(df)):
            idx = df.iloc[i]["ID_PARCEL"]
            if (self.input_folder / f"{int(idx) // 5000}/{idx}.npy").exists():
                valid_indices.append(i)

        df = df.iloc[valid_indices].reset_index(drop=True)
        self.logger.info(f"Filtered to {len(df)} rows with existing .npy files")

        # Save filtered dataset
        shapefile_path = output_path / "polygons_processed.shp"
        parquet_path = output_path / "polygons_processed.parquet"

        self.logger.info(
            f"Saving processed parcels to {shapefile_path} and {parquet_path}"
        )

        # Ensure output directory exists
        os.makedirs(output_path, exist_ok=True)

        # Save files
        df.to_file(shapefile_path, driver="ESRI Shapefile")
        df.to_parquet(parquet_path)

        return df

    def create_memmap(
        self,
        df: gpd.GeoDataFrame,
        output_folder: str,
        logger: Logger,
    ) -> np.memmap:
        """
        Convert individual .npy files to a memory-mapped array.

        Args:
            df: GeoDataFrame with valid parcels
            output_folder: Folder to save memory-mapped files

        Returns:
            np.memmap: Memory-mapped array
        """
        self.logger.info("Starting memmap conversion")

        # Ensure output directory exists
        os.makedirs(output_folder, exist_ok=True)

        # Function to get array by index
        def get_array(i):
            idx = df.iloc[i]["ID_PARCEL"]
            return np.load(self.input_folder / f"{int(idx) // 5000}/{idx}.npy")

        # Create memory-mapped array from generator
        from tqdm.autonotebook import tqdm

        self.logger.info(f"Creating memory-mapped array from {len(df)} parcels")
        mmap_ninja.np_from_generator(
            out_dir=output_folder,
            sample_generator=map(get_array, tqdm(range(len(df)))),
            batch_size=16384,
        )

        # Open and verify the created memory-mapped array
        memmap = mmap_ninja.np_open_existing(output_folder)
        self.logger.info(f"Memmap conversion completed. Final shape: {memmap.shape}")

        return memmap

    def clear_folder(self, folder_path: Path):
        if not folder_path.exists():
            self.logger.warning(f"Folder {folder_path} does not exist")
            return

        self.logger.info(f"Clearing folder: {folder_path}")
        for item in folder_path.iterdir():
            if item.is_file():
                item.unlink()
            elif item.is_dir():
                self.clear_folder(Path(item))
        os.rmdir(folder_path)


    def reset_folders(self, data: dict, logger: Logger):
        logger.info(f"Resetting folders: {data.folders_to_reset}")
        for folder in data.folders_to_reset:
            self.clear_folder(Path(folder))