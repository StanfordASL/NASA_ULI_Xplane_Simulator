# Getting Set Up with X-Plane 11
The following steps walk you through the steps required to set up X-Plane 11 for use with the code in this repository.

## Step 1: Download and Install X-Plane 11
X-Plane 11 can be purchased and downloaded [here](https://www.x-plane.com/desktop/buy-it/). It requires a $60 license to run.

## Step 2: Install the X-Camera Plugin
The X-Camera plugin can be downloaded [here](https://www.stickandrudderstudios.com/x-camera/download-x-camera/). Download it and follow the installation instructions. This plugin allows us to interface with the taxinet camera.

## Step 3: Download the 208B Grand Caravan Aircraft
1. The model can be purchased and downloaded [here](https://store.x-plane.org/C208B-GRAND-CARAVAN-HD-SERIES-XP11_p_668.html). 
2. Once downloaded, you should have a folder called `Carenado C208B_Grand_Caravan_v1.1`. Navigate to the `X-Plane 11` folder on your computer. From there, navigate to `Aircraft/Extra Aircraft` and drag the `Carenado C208B_Grand_Caravan_v1.1` folder to this location.

## Step 4: Download X-Plane Connect
To control the aircraft from python, we need to install [NASA's X-Plane Connect Plugin](https://github.com/nasa/XPlaneConnect).
1. Download the stable release (Version 1.2.1) [here](https://github.com/nasa/XPlaneConnect/releases).
2. Navigate to `X-Plane 11/Resources/plugins` and copy the contents of the downloaded folder to this location.

# Quick Start Tutorial
Next, let's do a quick tutorial to get you started on using X-Plane 11 and the python interface. This tutorial was written by Sydney Katz based on a tutorial originally written by Kyle Julian.

## Step 1: Open X-Plane 11 and Configure a Flight
1. Open X-Plane 11 (`X-Plane 11/X-Plane_x86_64`).
2. Select "New Flight".
3. In the AIRCRAFT panel, make sure the box next to "Show extra aircraft from old versions" in the upper right corner is checked.
4. Under General Aviation, you should see an option for "208B Grand Caravan". Select it.
5. In the LOCATION panel, select Grant Co Intl under AIRPORT NAME (ID is KMWH).
6. Optionally select weather and time of day (okay to just leave as is for this tutorial).
7. Click the "Start Flight" button in the bottom right corner.

It may take a minute or so to load in the flight. Once it loads, you should see a view from the cockpit of the aicraft sitting at the start of the runway. Next, we will configure X-Plane 11 to give us a view from the right wing of the aircraft using the X-Camera plugin.

## Step 2: Configure X-Camera
1. Click on the "Plugins" tab at the top of the screen.
2. From there hover over "X-Camera" and select "Toggle Control Panel" from the dropdown menu.
3. In the Camera panel located in the top right, scroll until you see "TaxiNet". Select it. The camera view should switch to the right wing of the aircraft.
    * NOTE: If TaxiNet is not in the Camera panel, the config file must be loaded. To select a camera config, go to File > Browse Community Aircraft Files, search for CameraSetting_X-Camera_Car_C208B.csv, and click "Import File".
4. Once you have the correct camera view, you can close the X-Camera Control Panel

We are now ready to start calling X-Plane 11 from python!

## Step 3: Connect to X-Plane 11 from Python
To control the aircraft from python, we can use [NASA's X-Plane Connect Plugin](https://github.com/nasa/XPlaneConnect). 

1. Open a terminal, navigate to `NASA_ULI_Xplane_Simulator/src`, and start python.
    ```shell script
    python3
    ```
2. Now that you have python open, let's import the functions we will need. The file `xpc3.py` was copied from the [XPlaneConnect repository](https://github.com/nasa/XPlaneConnect). The file `xpc3_helper.py` was written by us to provide some useful functions for the taxiing problem.
    ```python
    from xpc3 import *
    from xpc3_helper import *
    ```
3. We are now ready to connect to X-Plane 11. Make sure you have X-Plane 11 running and have followed all previous steps before running this line.
    ```python
    # Create connection to simulation
    # Simulation must be running and should not be in any settings menus
    client = XPlaneConnect()
    ```
4. Next, let's reset the simulation. This command will be helpful to run before starting any simulation episodes. It starts the aircraft at the beginning of the runway, in the center, and traveling at 5 m/s. However, the simulation will be paused, so the aircraft will not be moving forward yet.
    ```python
    reset(client)
    ```
5. We can now unpause the simulation.
    ```python
    client.pauseSim(False)
    ```
    After running this command, the aircraft should start to travel down the runway at 5 m/s. After watching it move, pause it again.
    ```python
        client.pauseSim(True)
    ```
6. We can control the aircraft by turning the nosewheel/rudder. The rudder input (fourth argument) should be between -1 and 1. Positive creates a right turn.
    ```python
    sendCTRL(client, 0.0, 0.0, 0.5, 0.0)

    # Unpause the simulation to observe the effect
    client.pauseSim(False)

    # Repause once done observing
    client.pauseSim(True)
    ```
7. In addition to controlling the aircraft, we can place it at a specified location on the runway. The runway is 20 meters wide and 2982 meters long.
    ```python
    # First let's reset it to the beginning of the runway
    reset(client)

    # Next, let's place it 5 meters ahead of our current location 
    # Use the setHomeState(client, x, y, theta function)
    # x is the distance from the centerline in meters (for runway -10 < x < 10, left is positive)
    # y is the distance down the runway in meters
    # theta is the angle that the aircraft faces measured from facing straight down the runway
    setHomeState(client, 0.0, 5.0, 0.0)

    # Let's move it to the left 4 meters
    setHomeState(client, 4.0, 5.0, 0.0)

    # Rotate it counterclockwise 10 degrees
    setHomeState(client, 4.0, 5.0, 10.0)

    # Move it to the end of the runway
    setHomeState(client, 0.0, 2982.0, 0.0)
    ```
8. X-Plane Connect provides a way to programmatically edit the [DataRefs](https://developer.x-plane.com/datarefs/) that control the simulation. We can easily get the value of a particular DataRef. For example, let's get the time of day (GMT) in seconds.
    ```python
    # Returns a tuple, so index the first entry to get the number back
    client.getDREF("sim/time/zulu_time_sec")[0]
    ```
9. We can also set the value of a DataRef. Let's set the time to 11:00PM local time.
    ```python
    client.sendDREF("sim/time/zulu_time_sec", 23 * 3600 + 8 * 3600)
    ```
    It should be nighttime now.

    DataRefs are simulation variables, and there are many many of them controlling everything from weather to aircraft position, speed, orientation, parking break, and more. They are summarized [here](https://developer.x-plane.com/datarefs/). Note that not all DataRefs can be written. Some DataRefs are calculated at each time step as a function of other DataRefs, so writing to them will have no effect because they will just be rewritten at the next time step.

This concludes the set up and tutorial! Check out the folders in this repo that use these functions to collect data from X-Plane 11 and simulate controllers.