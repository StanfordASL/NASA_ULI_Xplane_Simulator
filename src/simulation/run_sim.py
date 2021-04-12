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
        if settings.DUBINS:
            simulate_controller_dubins(client, settings.START_CTE, settings.START_HE,
                                settings.START_DTP, settings.END_DTP, settings.GET_STATE, settings.GET_CONTROL,
                                settings.DT, settings.CTRL_EVERY)
        else:
            simulate_controller(client, settings.START_CTE, settings.START_HE,
                                settings.START_DTP, settings.END_DTP, settings.GET_STATE, settings.GET_CONTROL)

def simulate_controller(client, startCTE, startHE, startDTP, endDTP, getState, getControl, simSpeed = 1.0):
    """ Simulates a controller using the built-in X-Plane 11 dynamics

        Args:
            client: XPlane Client
            startCTE: Starting crosstrack error (meters)
            startHE: Starting heading error (degrees)
            startDTP: Starting downtrack position (meters)
            endDTP: Ending downtrack position (meters)
            getState: Function to estimate the current crosstrack and heading errors.
                      Takes in an XPlane client and returns the crosstrack and
                      heading error estimates
            getControl: Function to perform control based on the state
                        Takes in an XPlane client, the current crosstrack error estimate,
                        and the current heading error estimate and returns a control effort
            -------------------
            simSpeed: increase beyond 1 to speed up simulation
    """
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
        # Deal with speed
        speed = xpc3_helper.getSpeed(client)
        throttle = 0.1
        if speed > 5:
            throttle = 0.0
        elif speed < 3:
            throttle = 0.2

        cte, he = getState(client)
        rudder = getControl(client, cte, he)
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


def dynamics(x, y, theta, phi_deg, dt=0.05, v=5, L=5):
    """ Dubin's car dynamics model (returns next state)

        Args:
            x: current crosstrack error (meters)
            y: current downtrack position (meters)
            theta: current heading error (degrees)
            phi_deg: steering angle input (degrees)
            -------------------------------
            dt: time step (seconds)
            v: speed (m/s)
            L: distance between front and back wheels (meters)
    """
    
    theta_rad = np.deg2rad(theta)
    phi_rad = np.deg2rad(phi_deg)

    x_dot = v * np.sin(theta_rad)
    y_dot = v * np.cos(theta_rad)
    theta_dot = (v / L) * np.tan(phi_rad)

    x_prime = x + x_dot * dt
    y_prime = y + y_dot * dt
    theta_prime = theta + np.rad2deg(theta_dot) * dt

    return x_prime, theta_prime, y_prime


def simulate_controller_dubins(client, startCTE, startHE, startDTP, endDTP, getState, getControl,
                               dt, ctrlEvery, simSpeed=1.0):
    """ Simulates a controller, overriding the built-in XPlane-11 dynamics to model the aircraft
        as a Dubin's car

        Args:
            client: XPlane Client
            startCTE: Starting crosstrack error (meters)
            startHE: Starting heading error (degrees)
            startDTP: Starting downtrack position (meters)
            endDTP: Ending downtrack position (meters)
            getState: Function to estimate the current crosstrack and heading errors.
                      Takes in an XPlane client and returns the crosstrack and
                      heading error estimates
            getControl: Function to perform control based on the state
                        Takes in an XPlane client, the current crosstrack error estimate,
                        and the current heading error estimate and returns a control effort
            dt: time step (seconds)
            crtlEvery: Frequency to get new control input 
                       (e.g. if dt=0.5, a value of 20 for ctrlEvery will perform control 
                       at a 1 Hz rate)
            -------------------
            simSpeed: increase beyond 1 to speed up simulation
    """
    # Reset to the desired starting position
    client.sendDREF("sim/time/sim_speed", simSpeed)
    xpc3_helper.reset(client, cteInit=startCTE,
                      heInit=startHE, dtpInit=startDTP)
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
