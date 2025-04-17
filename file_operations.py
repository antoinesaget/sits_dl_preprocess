#!/usr/bin/env python3

import mmap_ninja
import os
import numpy as np

class FileManager:

    def __init__(self):
        pass

    def filter_and_save_valid_parcels(self, df, input_folder, output_path, logger):
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
            if (input_folder / f"{int(idx) // 5000}/{idx}.npy").exists():
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

    def create_memmap(self, df, input_folder, output_folder, logger):
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
            return np.load(input_folder / f"{int(idx) // 5000}/{idx}.npy")

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
