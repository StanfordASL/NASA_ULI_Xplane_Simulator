# Data Generation
**NOTE:** the image gathering code in this repository is meant for a monitor with resolution 1920x1080 running in full screen mode. You can change back and forth from full screen mode is X-Plane 11 by opening the settings (second from the left in the upper right toolbar), opening the Graphics tab, and modifying the Monitor usage option under MONITER CONFIGURATION. For any other monitor configuration, the code would need to be modified. To navigate between terminals and X-Plane 11 when it is in full screen mode, use the `ALT + TAB` keyboard shortcut. 

The data generation code consists of three python files. If you would like to specify some settings and collect data, the main file you will want to familiarize yourself with is `settings.py`, which controls the settings used in the other two files. The file `sinusoidal.py` contains the code to send the aircraft down the runway in a sinusoidal pattern, and the file `data_recorder.py` saves screenshots and non-image data as the trajectories are run.

## Quick Start Tutorial
This tutorial assumes that you are already followed the steps [here](..) and that X-Plane 11 is currently running in the proper configuration for taxinet.

1. Modify the settings file based on your desired simulation parameters

    Modifiable parameters:
    
    `OUT_DIR`
    * Directory to save output data
    * NOTE: CSV file and images will be overwritten if already exists in that directory, but extra images (for time steps that do not occur in the new episodes) will not be deleted

    `TIME_OF_DAY`
    * Time of day in local time, e.g. 8.0 = 8AM, 17.0 = 5PM

    `CLOUD_COVER`
    * Cloud cover (higher numbers are cloudier/darker)
    * 0 = Clear, 1 = Cirrus, 2 = Scattered, 3 = Broken, 4 = Overcast

    `case`
    * Cases to run (determines how other variables are set)
    * `example`  - runs 2 short trajectories (used for initial testing)
    * `smallset` - runs 5 sinusoidal trajectories centered at zero crosstrack error with varying amplitude and frequency (ideal for collecting OoD data)
    * `largeset` - runs 20 sinusoidal trajectories with varying amplitude and frequency and centered at different crosstrack errors
    * `validation` - runs 5 sinusoidal trajectories centered at zero crosstrack error with 
    varying amplitude and frequency
    * `test` - runs 3 sinusoidal trajectories center at 3 different crosstrack errors
    * the last five trajectories of the largeset have the same parameter settings as the largeset

    `FREQUENCY`
    * Frequency with which to record data (in Hz)
    * NOTE: this is approximate due to computational overhead

2. Open two separate terminals and navigate to `NASA_ULI_Xplane_Simulator/src/data_generation` in each of them.

3. In the first terminal, run
    ```shell script
    python3 sinusoidal.py
    ```

4. Quickly after, in the second terminal, run
    ```shell script
    python3 data_recorder.py
    ```

5. Quickly minimize both terminals so that they are not in the way of the data recorder. There should be a five second buffer since starting the `sinusoidal.py` script.

6. Once all episodes are finished, the data will be saved to the folder you specified in `settings.py`.