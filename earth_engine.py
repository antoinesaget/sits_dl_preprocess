#!/usr/bin/env python3

import time
from logging import Logger

import ee
import pandas as pd
import shapely

from data_processing import DataProcessor


class EarthEngineClient:
    def __init__(self, logger: Logger, config: dict, all_bands: list):
        """
        Initialize the EarthEngineClient class.
        This class is responsible for interacting with Google Earth Engine
        Args:
            logger: Logger object for recording status
            config: Configuration dictionary
            all_bands: List of bands to retrieve from the satellite imagery
        """
        self.logger = logger
        self.config = config
        self.all_bands = all_bands

    def initialize_earth_engine(self, project_name: str) -> None:
        """
        Initialize Google Earth Engine with authentication if needed.

        Args:
            project_name: Google Earth Engine project name
        """
        try:
            ee.Initialize(
                opt_url="https://earthengine-highvolume.googleapis.com",
                project=project_name,
            )
            self.logger.info("Earth Engine initialized successfully")
        except Exception as e:
            self.logger.warning(
                f"Initial EE initialization failed, attempting authentication: {e}"
            )
            try:
                ee.Authenticate()
                ee.Initialize(
                    opt_url="https://earthengine-highvolume.googleapis.com",
                    project=project_name,
                )
                self.logger.info("Earth Engine initialized after authentication")
            except Exception as e:
                self.logger.error(
                    f"Failed to initialize Earth Engine after authentication: {e}"
                )
                raise

    def shapely2ee(self, geometry: shapely.Geometry) -> ee.Geometry:
        """
        Convert Shapely geometry to Earth Engine geometry.

        Args:
            geometry: Shapely geometry object

        Returns:
            ee.Geometry: Earth Engine geometry object
        """
        pt_list = list(zip(*geometry.exterior.coords.xy))
        return ee.Geometry.Polygon(pt_list)

    def query(self, region: ee.Geometry, start: str, end: str) -> list:
        """
        Query Earth Engine for satellite imagery.

        Args:
            region: Earth Engine geometry
            start: Start date (YYYY-MM-DD)
            end: End date (YYYY-MM-DD)

        Returns:
            List of image data
        """
        images = (
            ee.ImageCollection(self.config.collection)
            .filterDate(start, end)
            .filterBounds(region)
            .filter(ee.Filter.eq("GENERAL_QUALITY", "PASSED"))
            .select(self.all_bands)
        )

        sampled_points = ee.FeatureCollection.randomPoints(
            **{"region": region, "points": 200, "seed": 42}
        )

        return images.getRegion(sampled_points, self.config.scale).getInfo()

    def retrieve_data(
        self, region: ee.Geometry, row: pd.DataFrame, processor: DataProcessor
    ) -> pd.DataFrame:
        """
        Retrieve satellite data from Earth Engine with automatic retry for large areas.

        Args:
            region: Earth Engine geometry
            row: DataFrame row containing parcel info
            processor: DataProcessor object

        Returns:
            pd.DataFrame: Downloaded satellite data
        """
        retry = False
        steps = self.config.steps
        parcel_id = row["ID_PARCEL"]

        while True:
            if steps < 4:
                self.logger.debug(
                    f"Parcel {parcel_id} too large to process. Skipping..."
                )
                break

            try:
                if retry:
                    # Try with smaller time windows
                    starts, ends = processor.get_time_windows(
                        self.config.start, self.config.end, steps
                    )
                    dataframe = pd.DataFrame()
                    self.logger.debug(
                        f"Retrying with {len(starts)} time steps for parcel {parcel_id}"
                    )

                    for _start, _end in zip(starts, ends):
                        getinfo_dict = self.query(region, _start, _end)
                        dataframe_local = processor.parse(getinfo_dict)
                        if len(dataframe_local) > 0:
                            dataframe = pd.concat([dataframe, dataframe_local])

                    return dataframe.reset_index(drop=True)
                else:
                    # Try with one large time window
                    getinfo_dict = self.query(
                        region, self.config.start, self.config.end
                    )
                    dataframe = processor.parse(getinfo_dict)
                    return dataframe

            except ee.ee_exception.EEException as e:
                if "ImageCollection.getRegion: Too many values:" in str(e):
                    if retry:
                        steps = steps // 2
                        self.logger.debug(
                            f"Reducing steps to {steps} for parcel {parcel_id}"
                        )
                    retry = True
                    continue

                if "Too Many Requests" in str(e):
                    self.logger.debug("Rate limit hit, waiting 1 second...")
                    time.sleep(1)
                    continue

                self.logger.warning(f"Earth Engine error for parcel {parcel_id}: {e}")
                break

            except Exception as e:
                self.logger.warning(
                    f"Unexpected error processing parcel {parcel_id}: {e}"
                )
                break

        return pd.DataFrame()
