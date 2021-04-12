# Defines settings for collecting training data by running sinusoidal trajectories
# Written by Sydney Katz (smkatz@stanford.edu)

import fully_observable
import tinytaxinet

""" 
Parameters to be specified by user
    - Change these parameters to determine the cases you want to gather data for
"""

# Directory to save output data
# NOTE: CSV file and images will be overwritten if already exists in that directory, but
# extra images (for time steps that do not occur in the new episodes) will not be deleted
OUT_DIR = "/scratch/smkatz/NASA_ULI/benchmark/overcast/"

# Time of day in local time, e.g. 8.0 = 8AM, 17.0 = 5PM
TIME_OF_DAY = 8.0

# Cloud cover (higher numbers are cloudier/darker)
# 0 = Clear, 1 = Cirrus, 2 = Scattered, 3 = Broken, 4 = Overcast
CLOUD_COVER = 0

START_CTE = 6.0

START_HE = 0.0

START_DTP = 322.0

END_DTP = 422.0

GET_CONTROL = fully_observable.getProportionalControlDubins

GET_STATE = tinytaxinet.getStateTinyTaxiNet

"""
Parameters for Dubin's Model
"""

DT = 0.05

CTRL_EVERY = 20