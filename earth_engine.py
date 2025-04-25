#!/usr/bin/env python3

from logging import Logger
import time

from data_processing import DataProcessor
import shapely
import ee
import pandas as pd


class EarthEngineClient:
    def __init__(self):
        """
        Initialize the EarthEngineClient class.
        This class is responsible for interacting with Google Earth Engine
        """
        pass

    def initialize_earth_engine(self, logger: Logger, project_name: str) -> None:
        """
        Initialize Google Earth Engine with authentication if needed.

        Args:
            logger: Logger object for recording status
            project_name: Google Earth Engine project name
        """
        try:
            ee.Initialize(
                opt_url="https://earthengine-highvolume.googleapis.com",
                project=project_name,
            )
            logger.info("Earth Engine initialized successfully")
        except Exception as e:
            logger.warning(
                f"Initial EE initialization failed, attempting authentication: {e}"
            )
            ee.Authenticate()
            ee.Initialize(
                opt_url="https://earthengine-highvolume.googleapis.com",
                project=project_name,
            )
            logger.info("Earth Engine initialized after authentication")

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

    def query(
        self,
        region: ee.Geometry,
        start: str,
        end: str,
        collection: str,
        scale: int,
        all_bands: list,
    ) -> list:
        """
        Query Earth Engine for satellite imagery.

        Args:
            region: Earth Engine geometry
            start: Start date (YYYY-MM-DD)
            end: End date (YYYY-MM-DD)
            collection: Earth Engine collection name
            scale: Pixel scale in meters
            all_bands: List of bands

        Returns:
            List of image data
        """
        images = (
            ee.ImageCollection(collection)
            .filterDate(start, end)
            .filterBounds(region)
            .filter(ee.Filter.eq("GENERAL_QUALITY", "PASSED"))
            .select(all_bands)
        )

        sampled_points = ee.FeatureCollection.randomPoints(
            **{"region": region, "points": 200, "seed": 42}
        )

        return images.getRegion(sampled_points, scale).getInfo()

    def retrieve_data(
        self,
        region: ee.Geometry,
        row: pd.DataFrame,
        config: dict,
        logger: Logger,
        processor: DataProcessor,
        all_bands: list,
    ) -> pd.DataFrame:
        """
        Retrieve satellite data from Earth Engine with automatic retry for large areas.

        Args:
            region: Earth Engine geometry
            row: DataFrame row containing parcel info
            config: Configuration dictionary
            logger: Logger object
            processor: DataProcessor object
            all_bands: List of bands to retrieve

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
                    starts, ends = processor.get_time_windows(
                        config["start"], config["end"], steps
                    )
                    dataframe = pd.DataFrame()
                    logger.debug(
                        f"Retrying with {len(starts)} time steps for parcel {parcel_id}"
                    )

                    for _start, _end in zip(starts, ends):
                        getinfo_dict = self.query(
                            region,
                            _start,
                            _end,
                            config["collection"],
                            config["scale"],
                            all_bands,
                        )
                        dataframe_local = processor.parse(
                            getinfo_dict, config["columns_types"], all_bands
                        )
                        if len(dataframe_local) > 0:
                            dataframe = pd.concat([dataframe, dataframe_local])

                    return dataframe.reset_index(drop=True)
                else:
                    # Try with one large time window
                    getinfo_dict = self.query(
                        region,
                        config["start"],
                        config["end"],
                        config["collection"],
                        config["scale"],
                        all_bands,
                    )
                    dataframe = processor.parse(
                        getinfo_dict, config["columns_types"], all_bands
                    )
                    return dataframe

            except ee.ee_exception.EEException as e:
                if "ImageCollection.getRegion: Too many values:" in str(e):
                    if retry:
                        steps = steps // 2
                        logger.debug(
                            f"Reducing steps to {steps} for parcel {parcel_id}"
                        )
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
