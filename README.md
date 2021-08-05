# NASA ULI Xplane Simulator
Dataset and Simulator for the NASA ULI Project

# Download Links, Citation, and Stanford Persistent URL
Please see [here](https://purl.stanford.edu/zz143mb4347) for a citation and links to GBs of data, which you should download and place in `NASA_DATA_DIR` as specified below.

For more information on the data, check out [this document](docs/NASA_ULI_dataset_format).

# System Requirements
First, export a system (bash) variable corresponding to where you have cloned this repo named `NASA_ULI_ROOT_DIR`. For example, in your bashrc:

`export NASA_ULI_ROOT_DIR={path to this directory on your machine}`

Second, export a system (bash) variable corresponding to where you have downloaded data named `NASA_DATA_DIR`. For example, in your bashrc:

`export NASA_DATA_DIR='path to where data is downloaded}`

This code was tested using Python 3.6.12. In general, any version of Python 3 should work.
See src/requirements.txt for specific packages.

# Quick Links
* [X-Plane 11 Set Up Instructions](src/)
* [Data Generation Instructions](src/data_generation)
* [Controller Simulation Instructions](src/simulation)

# Repository Structure
- `src`
    - Has the main code. See `src/examples` for a tutorial.

- `docs`
    - Detailed specification and documentation.

- `data`
    - Images and corresponding state information from XPlane. 
    - This is only a subset of data for testing, the rest should be under `NASA_DATA_DIR`.

- `scratch`
    - Create this folder on your machine, where results will be saved. Do not check this into the main GIT repository to keep it compact.
