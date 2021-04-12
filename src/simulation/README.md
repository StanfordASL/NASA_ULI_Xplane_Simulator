# Simulation
**NOTE:** the image-based controllers in this repository are designed for a monitor with resolution 1920x1080 running in full screen mode. You can change back and forth from full screen mode is X-Plane 11 by opening the settings (second from the left in the upper right toolbar), opening the Graphics tab, and modifying the Monitor usage option under MONITER CONFIGURATION. For any other monitor configuration, the code would need to be modified. To navigate between terminals and X-Plane 11 when it is in full screen mode, use the `ALT + TAB` keyboard shortcut. 

In this folder, we provide example implementations of proportional controllers in for the taxinet problem in X-Plane 11.

## Dynamics Models
We support two models of aircraft taxi dynamics: the dynamics built into the X-Plane 11 flight simulator and a simpler Dubin's car model of the aircraft.

### X-Plane 11 Dynamics
More information about the X-Plane 11 dynamics model can be found [here](https://www.x-plane.com/desktop/how-x-plane-works/). Control is performed using a rudder command that ranges between -1 and 1. Positive values create right turns, while negative values create left turns.

### Dubin's Car Dynamics
We also support a simpler model of the aircraft taxi dynamics by modeling it as a discrete time Dubin's car (see equation 13.5 [here](http://planning.cs.uiuc.edu/node658.html)). To do this, we repeatly calculate the next position independent of the X-Plane 11 dynamics and simply call a function to place the aircraft there. Control for this model is performed using a steering angle command (specified in degrees).

## Code Overview
The controller simulation code consists of two main python files. If you would like to specify some settings and run the controllers, the main file you will want to familiarize yourself with is `settings.py`, which controls the settings used in the simulation. The file `run_sim.py` contains the main code for running simulations.

The file `controllers.py` contains functions for proportional control, and the files `fully_observable.py` and `tiny_taxinet.py` contain state estimation functions. The `nnet.py` file has some necessary functions for working with neural networks in the .nnet format (more info [here](https://github.com/sisl/NNet)).

## Quick Start Tutorial
This tutorial assumes that you are already followed the steps [here](..) and that X-Plane 11 is currently running in the proper configuration for taxinet.

1. Modify the settings file based on your desired simulation parameters

    Modifiable parameters:
    
    `DUBINS`
    * Whether or not to override the X-Plane 11 simulator dynamics with a Dubin's car model
    * True = Use Dubin's car model, False = Use X-Plane 11 dynamics

    `STATE_ESTIMATOR`
    * Type of state estimation
    * 'fully_observable' - true state is known
    * 'tiny_taxinet' - state is estimated using the tiny taxinet neural network from
                           image observations of the true state

    `TIME_OF_DAY`
    * Time of day in local time, e.g. 8.0 = 8AM, 17.0 = 5PM

    `CLOUD_COVER`
    * Cloud cover (higher numbers are cloudier/darker)
    * 0 = Clear, 1 = Cirrus, 2 = Scattered, 3 = Broken, 4 = Overcast

    `START_CTE`
    * Starting crosstrack error in meters (for initial position on runway)

    `START_HE`
    * Starting heading error in degrees (for initial position on runway)

    `START_DTP`
    * Starting downtrack position in meters (for initial position on runway)

    `END_DTP`
    * Used to determine when to stop the simulation

2. Open a terminal and navigate to `NASA_ULI_Xplane_Simulator/src/simulation` in each of them.

3. In the terminal, run
```shell script
python3 run_sim.py
```

4. Quickly minimize the terminal (if using the image-based controller) so that it does not get in the way of the screenshots. There should be a five second buffer since starting the `run_sim.py` script.