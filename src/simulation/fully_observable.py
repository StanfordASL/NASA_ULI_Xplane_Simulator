import sys
import os

# make sure this is a system variable in your bashrc
NASA_ULI_ROOT_DIR = os.environ['NASA_ULI_ROOT_DIR']
XPC3_DIR = NASA_ULI_ROOT_DIR + '/src/'
sys.path.append(XPC3_DIR)

import xpc3_helper
import xpc3


def getStateFullyObservable(client):
    """ Returns the true crosstrack error (meters) and
        heading error (degrees) to simulate fully 
        oberservable control

        Args:
            client: XPlane Client
    """
    cte, _, he = xpc3_helper.getHomeState(client)
    return cte, he
