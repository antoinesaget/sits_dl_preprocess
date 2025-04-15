#!/usr/bin/env python3

from data_processing import DataProcessor
import ee
import time
import pandas as pd

from main import ALL_BANDS

class EarthEngineClient:
    
    def __init__(self, logger):
        self.logger = logger
        self.initialize_earth_engine()
        self.logger.info("Earth Engine client initialized")

    def initialize_earth_engine(self):
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
            self.logger.info("Earth Engine initialized successfully")
        except Exception as e:
            self.logger.warning(
                f"Initial EE initialization failed, attempting authentication: {e}"
            )
            ee.Authenticate()
            ee.Initialize(
                opt_url="https://earthengine-highvolume.googleapis.com",
                project="ee-magzoumov",
            )
            self.logger.info("Earth Engine initialized after authentication")

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

    def retrieve_data(self, region, row, config):
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
                self.logger.error(f"Parcel {parcel_id} too large to process. Skipping...")
                break

            try:
                processor = DataProcessor(self.logger)
                if retry:
                    # Try with smaller time windows
                    starts, ends = processor.get_time_windows(config["start"], config["end"], steps)
                    dataframe = pd.DataFrame()
                    self.logger.debug(
                        f"Retrying with {len(starts)} time steps for parcel {parcel_id}"
                    )

                    for _start, _end in zip(starts, ends):
                        getinfo_dict = self.query(
                            region, _start, _end, config["collection"], config["scale"]
                        )
                        dataframe_local = processor.parse(getinfo_dict, config["columns_types"])
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
                    )
                    dataframe = processor.parse(getinfo_dict, config["columns_types"])
                    return dataframe

            except ee.ee_exception.EEException as e:
                if "ImageCollection.getRegion: Too many values:" in str(e):
                    if retry:
                        steps = steps // 2
                        self.logger.debug(f"Reducing steps to {steps} for parcel {parcel_id}")
                    retry = True
                    continue

                if "Too Many Requests" in str(e):
                    self.logger.debug("Rate limit hit, waiting 1 second...")
                    time.sleep(1)
                    continue

                self.logger.error(f"Earth Engine error for parcel {parcel_id}: {e}")
                break

            except Exception as e:
                self.logger.error(f"Unexpected error processing parcel {parcel_id}: {e}")
                break

        return pd.DataFrame()
