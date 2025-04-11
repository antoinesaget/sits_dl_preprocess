# Satellite Imagery Download and Processing Pipeline

This repository contains a data processing pipeline designed to download Sentinel-2 satellite imagery from Google Earth Engine, process it into a standardized format, and save it as memory-mapped arrays for efficient access. The pipeline is specifically designed for working with agricultural parcels in France.

## Overview

The main script `sits_dl_preprocess.py` performs the following operations:

1. Downloads Sentinel-2 satellite imagery from Google Earth Engine using the Earth Engine API
2. Processes and filters the imagery data
3. Converts processed data into NumPy arrays
4. Organizes data into memory-mapped files for efficient access

## Prerequisites

### Required Packages

- Earth Engine API (`ee`)
- GeoPandas (`geopandas`)
- NumPy (`numpy`)
- Pandas (`pandas`)
- mmap-ninja (`mmap_ninja`)
- Rich (`rich`) for progress tracking
- tqdm for progress visualization

### Authentication

- Google Earth Engine account with authentication set up
- The script uses the project `ee-antoinesaget` - you may need to modify this for your own use

## Installation

1. Clone this repository:
   ```
   git clone <repository-url>
   cd <repository-directory>
   ```

2. Create and activate the conda environment:
   ```
   conda env create -f environment.yml
   conda activate sits_dl
   ```

3. Set up Earth Engine authentication:
   ```
   earthengine authenticate
   ```

## Usage

The main script can be executed directly:

```
python sits_dl_preprocess.py
```

### Configuration

The script uses a default configuration defined in `DEFAULT_CONFIG`:

```python
DEFAULT_CONFIG = {
    'start': '2022-01-01',
    'end': '2022-12-31',
    'collection': 'COPERNICUS/S2_SR_HARMONIZED',
    'scale': 10,
    'columns_types': {
        'SCL': 'category',
        'MSK_CLDPRB': 'int8',
        'B1': 'int16',
        # ... other bands
    },
    'steps': 64,
    'area_min': 0.1,  # Minimum parcel area in hectares
    'area_max': 40,  # Maximum parcel area in hectares
}
```

You can modify these parameters in the script before running it.

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

- `sits_dl_preprocess.py`: Main script for downloading and processing satellite imagery
- `sample_agricultural_parcels_10k.parquet`: Sample dataset containing agricultural parcels
- `environment.yml`: Conda environment configuration
- `processed_arrays/`: Directory containing individual processed NumPy arrays (created during execution)
- `out_memmap/`: Directory containing memory-mapped arrays for efficient access (created during execution)

## Areas for Improvement

As an intern working on this codebase, here are some suggested areas for improvement:

1. **Code Structure**: Refactor the monolithic script into a modular package structure
   - Create separate modules for Earth Engine interactions, data processing, and file operations
   - Implement a proper class hierarchy

2. **Configuration Management**: 
   - Move configuration from hardcoded dictionary to external YAML/JSON config file
   - Add command-line argument parsing for flexible execution

3. **Error Handling and Logging**:
   - Improve error handling with more specific exception catches
   - Enhance logging with more detailed progress information
   - Add data validation steps

4. **Performance Optimization**:
   - Review parallel processing implementation
   - Optimize memory usage

5. **Documentation**:
   - Add in-line documentation following standard conventions
   - Add usage examples

6. **User Interface**:
   - Add a simple CLI interface

7. **Packaging**:
   - Document dependencies properly

## Contributing

To contribute to this project:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

[TODO]
