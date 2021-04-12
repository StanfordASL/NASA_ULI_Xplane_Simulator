import settings
import cv2
import mss
import time
import numpy as np
import sys
import os

# make sure this is a system variable in your bashrc
NASA_ULI_ROOT_DIR = os.environ['NASA_ULI_ROOT_DIR']
XPC3_DIR = NASA_ULI_ROOT_DIR + '/src/'
sys.path.append(XPC3_DIR)

import xpc3_helper
import xpc3

def main():
    with xpc3.XPlaneConnect() as client:
        # Set weather and time of day
        client.sendDREF("sim/time/zulu_time_sec", settings.TIME_OF_DAY * 3600 + 8 * 3600)
        client.sendDREF("sim/weather/cloud_type[0]", settings.CLOUD_COVER)

        # Run the simulation
        simulate_controller(client, settings.START_CTE, settings.START_HE,
                            settings.START_DTP, settings.END_DTP, settings.GET_STATE, settings.GET_CONTROL)

def simulate_controller(client, startCTE, startHE, startDTP, endDTP, getState, getControl, simSpeed = 1.0):
    # Reset to the desired starting position
    client.sendDREF("sim/time/sim_speed", simSpeed)
    xpc3_helper.reset(client, cteInit = startCTE, heInit = startHE, dtpInit = startDTP)
    xpc3_helper.sendBrake(client, 0)

    time.sleep(5)  # 5 seconds to get terminal window out of the way
    client.pauseSim(False)

    dtp = startDTP
    startTime = client.getDREF("sim/time/zulu_time_sec")[0]
    endTime = startTime

    while dtp < endDTP:
        cte, he = getState(client)
        throttle, rudder = getControl(client, cte, he)
        client.sendCTRL([0, rudder, rudder, throttle])

        # Wait for next timestep
        while endTime - startTime < 1:
            endTime = client.getDREF("sim/time/zulu_time_sec")[0]
            time.sleep(0.001)

        # Set things for next round
        startTime = client.getDREF("sim/time/zulu_time_sec")[0]
        endTime = startTime
        _, dtp, _ = xpc3_helper.getHomeState(client)
        time.sleep(0.001)

    client.pauseSim(True)

if __name__ == "__main__":
    main()
