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
                            settings.START_DTP, settings.END_DTP, settings.GET_STATE, settings.GET_CONTROL,
                            settings.DT, settings.CTRL_EVERY)


def dynamics(x, y, theta, phi_deg, dt=0.05, v=5, L=5):
    theta_rad = np.deg2rad(theta)
    phi_rad = np.deg2rad(phi_deg)

    x_dot = v * np.sin(theta_rad)
    y_dot = v * np.cos(theta_rad)
    theta_dot = (v / L) * np.tan(phi_rad)

    x_prime = x + x_dot * dt
    y_prime = y + y_dot * dt
    theta_prime = theta + np.rad2deg(theta_dot) * dt

    return x_prime, theta_prime, y_prime


def simulate_controller(client, startCTE, startHE, startDTP, endDTP, getState, getControl, 
                        dt, ctrlEvery, simSpeed = 1.0):
    # Reset to the desired starting position
    client.sendDREF("sim/time/sim_speed", simSpeed)
    xpc3_helper.reset(client, cteInit = startCTE, heInit = startHE, dtpInit = startDTP)
    xpc3_helper.sendBrake(client, 0)

    time.sleep(5)  # 5 seconds to get terminal window out of the way

    cte = startCTE
    he = startHE
    dtp = startDTP
    startTime = client.getDREF("sim/time/zulu_time_sec")[0]
    endTime = startTime

    while dtp < endDTP:
        cte_pred, he_pred = getState(client)
        phiDeg = getControl(client, cte_pred, he_pred)

        for i in range(ctrlEvery):
            cte, he, dtp = dynamics(cte, dtp, he, phiDeg, dt)
            xpc3_helper.setHomeState(client, cte, dtp, he)
            time.sleep(0.03)


if __name__ == "__main__":
    main()
