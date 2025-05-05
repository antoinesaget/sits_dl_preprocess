# Satellite Imagery Download and Processing Pipeline

This repository contains a data processing pipeline designed to download Sentinel-2 satellite time series from Google Earth Engine, process it into a standardized format, and save it as memory-mapped arrays for efficient access. 
It takes as input a parquet file containing polygons and outputs a memory-mapped array containing the downloaded and processed data for each polygon.

## Overview

The main script `main.py` performs the following operations:

1. Downloads Sentinel-2 satellite imagery from Google Earth Engine using the Earth Engine API
2. Processes and filters the imagery data
3. Converts processed data into NumPy arrays
4. Organizes data into memory-mapped files for efficient access

## Installation

### 1. Clone repository

```
git clone <repository-url>
cd <repository-directory>
```

### 2. Create python environnement

- Install conda/mamba from [here](https://github.com/conda-forge/miniforge)
- Create and activate the conda environment (creating the environment is only needed once, the next time you will just need to do conda activate sits_dl):
   ```
   conda env create -f environment.yaml
   conda activate sits_dl
   ```

**Dependencies:**

- Earth Engine API (`ee`)
- GeoPandas (`geopandas`)
- NumPy (`numpy`)
- Pandas (`pandas`)
- mmap-ninja (`mmap_ninja`)
- Pyyaml (`yaml`)
- Argparse (`argparse`)
- Rich (`rich`) for progress tracking
- tqdm for progress visualization


### 3. Authenticate with Earth Engine

- You need a Google Earth Engine account with authentication set up
- You need to specify your own earth engine project in order to use the script. You can do that in the [config file](config.yaml) directly, or from the command line (default.ee_project_name=your-project)
- Authenticate : `earthengine authenticate`


## Usage

The main script can be executed directly (after activating the conda environment sits_dl created previously):

```
python main.py
```

### Configuration

The script uses a default configuration defined in [config.yaml](config.yaml):

```yaml
default:
  start: "2022-01-01" # Start date for the analysis, format YYYY-MM-DD
  end: "2022-12-31" # End date for the analysis, format YYYY-MM-DD
  collection: COPERNICUS/S2_SR_HARMONIZED
  scale: 10
  columns_types:
    SCL: category
    MSK_CLDPRB: int8
    B1: int16
    B2: int16
    B3: int16
    B4: int16
    B5: int16
    B6: int16
    B7: int16
    B8: int16
    B8A: int16
    B9: int16
    B11: int16
    B12: int16
  steps: 64
  area_min: 0.1  # Minimum parcel area in hectares
  area_max: 40  # Maximum parcel area in hectares
  filter_start: "2022-02-01" # Start date for filtering, format YYYY-MM-DD
  filter_end: "2022-11-30" # End date for filtering, format YYYY-MM-DD
  ee_project_name: "ee-magzoumov" # Earth Engine project name
  days_interval: 5 # Sample every 5th date
  points: 100 # Number of points for each parcel
```

You can modify these parameters before running the script.

### Input Data

The script expects:
- A parquet file containing GeoDataFrame with agricultural parcel polygons
- Each parcel should have a unique identifier

The repository includes a sample parquet file (`sample_agricultural_parcels_10k.parquet`) with 10,000 agricultural parcels that can be used for testing.

### Output Data

The script produces:
- Individual `.npy` files for each parcel (raw processed data) in the `processed_arrays` directory
- A memory-mapped array containing all processed data in the `out_memmap` directory
- Filtered dataset files (shapefile and parquet) containing only valid parcels

## Project Structure

- [`main.py`](main.py): Main script for downloading and processing satellite imagery
- [`data_processing.py`](data_processing.py): data processing methods (cloud filtering, alignment, etc.)
- [`earth_engine.py`](earth_engine.py): Earth Engine interaction methods (querying, downloading, etc.)
- [`file_operations.py`](file_operations.py): file operations methods (saving, conversion to memory-mapped arrays)
- [`sample_agricultural_parcels_10k.parquet`](sample_agricultural_parcels_10k.parquet): Sample dataset containing agricultural parcels
- [`environment.yaml`](environment.yaml): Conda environment configuration
- `processed_arrays/`: Directory containing individual processed NumPy arrays (created during execution)
- `out_memmap/`: Directory containing memory-mapped arrays for efficient access (created during execution)

## Areas for Improvement

Suggested areas for improvement:

1. [x] **Code Structure**: Refactor the monolithic script into a modular package structure
   - Create separate modules for Earth Engine interactions, data processing, and file operations
   - Implement a proper class hierarchy

2. [v] **Configuration Management**: 
   - Move configuration from hardcoded dictionary to external YAML/JSON config file
   - Add command-line argument parsing for flexible execution

3. [ ] **Error Handling and Logging**:
   - Improve error handling with more specific exception catches
   - Enhance logging with more detailed progress information
   - Add data validation steps

4. [ ] **Performance Optimization**:
   - Review parallel processing implementation
   - Optimize memory usage

5. [ ] **Documentation**:
   - Add in-line documentation following standard conventions
   - Add usage examples

6. [x] **User Interface**: add a simple CLI interface

7. [x] **Packaging**: document dependencies properly

## Contributing

To contribute to this project:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is open source and available under the MIT License.
