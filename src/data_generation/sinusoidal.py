import sys
import os

# make sure this is a system variable in your bashrc
NASA_ULI_ROOT_DIR=os.environ['NASA_ULI_ROOT_DIR']
XPC3_DIR = NASA_ULI_ROOT_DIR + '/src/'
sys.path.append(XPC3_DIR)

import numpy as np
import xpc3
import xpc3_helper
import time

import mss
import cv2

import settings

def main():
    with xpc3.XPlaneConnect() as client:
        # Set weather and time of day
        client.sendDREF("sim/time/zulu_time_sec", settings.TIME_OF_DAY * 3600 + 8 * 3600)
        client.sendDREF("sim/weather/cloud_type[0]", settings.CLOUD_COVER)
        
        # Run the trajectories
        runTrainingCases(client, settings.CASE_INDS, endPerc = settings.END_PERC)

def runSinusoidal(client, headingLimit, turn, centerCTE, simSpeed = 1.0, endPerc = 99):
    """ Runs a sinusoidal trajectory down the runway

        Args:
            client: XPlane Client
            headingLimit: max degrees aircraft heading direction will deviate from 
                          runway direction (might go slightly past this)
            turn: dictates rudder/nosewheel strength (gain on the rudder/nosewheel command)
                  Larger value means tighter turns
            centerCTE: center of sinusoidal trajectory (in meters from centerline)
            -------------------
            simSpeed: increase beyond 1 to speed up simulation
            endPerc: percentage down runway to end trajectory and reset
    """

    # Reset to beginning of runway
    client.sendDREF("sim/time/sim_speed", simSpeed)
    xpc3_helper.reset(client)
    xpc3_helper.sendBrake(client, 0)

    time.sleep(5) # 5 seconds to get terminal window out of the way
    client.pauseSim(False)

    while xpc3_helper.getPercDownRunway(client) < endPerc:
        getSinusoidalControl(client, headingLimit, turn, centerCTE)
        time.sleep(0.03)
    
    xpc3_helper.reset(client)

def getSinusoidalControl(client, headingLimit, turn, centerCTE):
    """ Applies control necessary to continue with sinusoidal trajectory

        Args:
            client: XPlane Client
            headingLimit: Max degrees aircraft heading direction will deviate from 
                          runway direction (might go slightly past this)
            turn: Dictates rudder/nosewheel strength (gain on the rudder/nosewheel command)
                  Larger value means tighter turns
            centerCTE: Center of sinusoidal trajectory (in meters from centerline)
    """
    
    speed = xpc3_helper.getSpeed(client)
    cte, _, he = xpc3_helper.getHomeState(client)

    # Deal with speed
    throttle = 0.1
    if speed > 5:
        throttle = 0.0
    elif speed < 3:
        throttle = 0.2

    # Amount of rudder needed to go straight, roughly
    rudder = 0.008
    if he < headingLimit and cte < centerCTE:
        rudder -= turn
    elif he > -headingLimit and cte > centerCTE:
        rudder += turn

    xpc3_helper.sendCTRL(client, 0, rudder, rudder, throttle)

""" Various training cases """

def getParams(num):
    """ Returns the set of parameters for a given sinusoidal trajectory

        Args:
            num: case number to get the parameters for
    """
    angLimit = 5
    turn = 0.05
    centerCTE = 0

    if num < 7:
        centerCTE = np.linspace(-8, 8, 7)[num]
    elif num < 15:
        angLimit = 10
        if num >= 11:
            turn = 0.2
        centerCTE = np.linspace(-6, 6, 4)[(num-7) % 4]
    elif num < 18:
        angLimit = 20
        turn = [0.1, 0.2, 0.3][num-15]
    elif num < 20:
        angLimit = 25
        turn = [0.25, 0.35][num-18]
    elif num < 23:
        angLimit = 20
        turn = [0.08, 0.18, 0.28][num-20]
    elif num < 25:
        angLimit = 25
        turn = [0.23, 0.33][num-23]
    else:
        angLimit = 10
        turn = 0.1
        centerCTE = (num-26)*3
    return angLimit, turn, centerCTE

def runTrainingCases(client, caseNums, endPerc = 99):
    """ Main function to run the training cases

        Args:
            client: XPlane Client
            caseNums: indices of cases to run
            -------------------
            endPerc: percentage down runway to end trajectory and reset
    """
    for i in caseNums:
        angLimit, turn, centerCTE = getParams(i)
        runSinusoidal(client, angLimit, turn, centerCTE, endPerc=endPerc)
        time.sleep(1)

if __name__ == "__main__":
    main()
