import numpy as np

def getProportionalControl(client, cte, he):
    """ Returns rudder command using proportional control
        for use with X-Plane 11 dynamics

        Args:
            client: XPlane Client
            cte: current estimate of the crosstrack error (meters)
            he: current estimate of the heading error (degrees)
    """
    # Amount of rudder needed to go straight, roughly.
    rudder = 0.008  # 0.004

    # Proportional controller
    cteGain = 0.015
    heGain = 0.008
    rudder += np.clip(cteGain * cte + heGain * he, -1.0, 1.0)
    return rudder

def getProportionalControlDubins(client, cte, he):
    """ Returns steering angle command using proportional control
        for use with Dubin's car dynamics

        Args:
            client: XPlane Client
            cte: current estimate of the crosstrack error (meters)
            he: current estimate of the heading error (degrees)
    """
    return -0.74 * cte - 0.44 * he
