import numpy as np
import sys
import os

# make sure this is a system variable in your bashrc
NASA_ULI_ROOT_DIR = os.environ['NASA_ULI_ROOT_DIR']
XPC3_DIR = NASA_ULI_ROOT_DIR + '/src/'
sys.path.append(XPC3_DIR)

import xpc3_helper
import xpc3

def getProportionalControl(client, cte, he):
    speed = xpc3_helper.getSpeed(client)

    # Deal with speed
    throttle = 0.1
    if speed > 5:
        throttle = 0.0
    elif speed < 3:
        throttle = 0.2

    # Amount of rudder needed to go straight, roughly.
    rudder = 0.008  # 0.004

    # Proportional controller
    cteGain = 0.015
    heGain = 0.008
    rudder += np.clip(cteGain * cte + heGain * he, -1.0, 1.0)
    return throttle, rudder

def getProportionalControlDubins(client, cte, he):
    return -0.74 * cte - 0.44 * he


def getStateFullyObservable(client):
    cte, _, he = xpc3_helper.getHomeState(client)
    return cte, he
