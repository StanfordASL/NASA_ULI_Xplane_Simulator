# Defines settings for collecting training data by running sinusoidal trajectories
# Written by Sydney Katz (smkatz@stanford.edu)

import controllers
import fully_observable
import tiny_taxinet

""" 
Parameters to be specified by user
    - Change these parameters to determine the cases you want to gather data for
"""

# Whether or not to override the X-Plane 11 simulator dynamics with a Dubin's car model
# True = Use Dubin's car model
# False = Use X-Plane 11 dynamics
DUBINS = False

# Type of state estimation
# 'fully_observable' - true state is known
# 'tiny_taxinet'     - state is estimated using the tiny taxinet neural network from
#                      image observations of the true state
STATE_ESTIMATOR = 'fully_observable'

# Time of day in local time, e.g. 8.0 = 8AM, 17.0 = 5PM
TIME_OF_DAY = 8.0

# Cloud cover (higher numbers are cloudier/darker)
# 0 = Clear, 1 = Cirrus, 2 = Scattered, 3 = Broken, 4 = Overcast
CLOUD_COVER = 0

# Starting crosstrack error in meters
START_CTE = 6.0

# Starting heading error in degrees
START_HE = 0.0

# Starting downtrack position in meters
START_DTP = 322.0

# Downtrack positions (in meters) to end the simulation
END_DTP = 422.0

"""
Parameters for Dubin's Model
"""

# Time steps for the dynamics in seconds
DT = 0.05

# Frequency to get new control input 
# (e.g. if DT=0.5, CTRL_EVERY should be set to 20 to perform control at a 1 Hz rate)
CTRL_EVERY = 20

"""
Other parameters
    - NOTE: you should not need to change any of these unless you want to create
    additional scenarios beyond the ones provided
"""

# Tells simulator which proportional controller to use based on dynamics model
if DUBINS:
    GET_CONTROL = controllers.getProportionalControlDubins
else:
    GET_CONTROL = controllers.getProportionalControl

# Tells simulator which function to use to estimate the state
if STATE_ESTIMATOR == 'tiny_taxinet':
    GET_STATE = tiny_taxinet.getStateTinyTaxiNet
elif STATE_ESTIMATOR == 'fully_observable':
    GET_STATE = fully_observable.getStateFullyObservable
else:
    print("Invalid state estimator name - assuming fully observable")
    GET_STATE = fully_observable.getStateFullyObservable
