import sys
import os

# make sure this is a system variable in your bashrc
NASA_ULI_ROOT_DIR = os.environ['NASA_ULI_ROOT_DIR']
XPC3_DIR = NASA_ULI_ROOT_DIR + '/src/'
sys.path.append(XPC3_DIR)

import numpy as np
import xpc3
import xpc3_helper
import time
import os

import mss
import cv2

import settings

screenShot = mss.mss()

def main():
    with xpc3.XPlaneConnect() as client:
        record(client, settings.OUT_DIR, endPerc = settings.END_PERC - 1.0, 
                    freq=settings.FREQUENCY, numEpisodes=len(settings.CASE_INDS))

def record(client, outDir, startPerc = 1.5, endPerc = 10, freq = 10, numEpisodes = 1):
    """ Record data from training episodes in XPlane into a CSV and PNG files

        Args:
            client: XPlane Client
            outDir: output directory for csv file and png images
            -----------------
            startPerc: percentage down runway to start collecting data
            endPerc: percentage down runway to finish collecting data
                     (NOTE: this must be less than endPerc for the sinusoidal trajectory)
            freq: frequency to save data
                  (NOTE: this is approximate due to computational overhead)
            numEpisodes: number of episodes to record for
    """
    # Make data folder if it doesn't exist
    if not os.path.exists(outDir):
        os.makedirs(outDir)

    # Initialize the CSV file
    csvFile = outDir + 'labels.csv'
    with open(csvFile, 'w') as fd:
        fd.write('image_filename,absolute_time_GMT_seconds,relative_time_seconds,distance_to_centerline_meters,')
        fd.write('distance_to_centerline_NORMALIZED,downtrack_position_meters,downtrack_position_NORMALIZED,')
        fd.write('heading_error_degrees,heading_error_NORMALIZED,period_of_day,cloud_type\n')

    for i in range(numEpisodes):
        first = True
        percDownRunway = xpc3_helper.getPercDownRunway(client)
        currStep = 0

        while percDownRunway > endPerc:
            time.sleep(0.5)
            percDownRunway = xpc3_helper.getPercDownRunway(client)

        while percDownRunway < endPerc:
            if ((percDownRunway > startPerc) and (percDownRunway < 45.0)) or ((percDownRunway > 51.5) and (percDownRunway < endPerc)):
                if first:
                    startTime = client.getDREF("sim/time/zulu_time_sec")[0]
                    addCurrData(client, outDir, csvFile, startTime, currStep, i + 1)
                    first = False
                    time.sleep(1 / freq)
                else:
                    addCurrData(client, outDir, csvFile,
                                startTime, currStep, i + 1)
                    time.sleep(1 / freq)
                currStep += 1
            else:
                time.sleep(1)
            
            percDownRunway = xpc3_helper.getPercDownRunway(client)

def addCurrData(client, outDir, csvFile, startTime, currStep, episodeNum):
    """ Add current data to csv file and save off a screenshot

        Args:
            client: XPlane Client
            outDir: output directory for csv file and png images
            csvFile: name of the CSV file for non-image data (should be initialized already)
            startTime: absolute time that the episode started
            currStep: current step of saving data (for image labeling)
            episodeNum: number of the current episode (for image labeling)
    """
    # Time step information
    absolute_time = client.getDREF("sim/time/zulu_time_sec")[0]
    time_step = absolute_time - startTime

    # Plane state information
    cte, dtp, he = xpc3_helper.getHomeState(client)

    # Environmental conditions
    local_time = client.getDREF("sim/time/local_time_sec")[0]
    if local_time < 5 * 3600 or local_time > 17 * 3600:
        period_of_day = 2
        time_period = 'night'
    elif local_time > 12 * 3600 and local_time < 17 * 3600:
        period_of_day = 1
        time_period = 'afternoon'
    else:
        period_of_day = 0
        time_period = 'morning'

    cloud_cover = client.getDREF("sim/weather/cloud_type[0]")[0]
    weather = settings.WEATHER_TYPES[cloud_cover]

    # Image information
    img = cv2.cvtColor(np.array(screenShot.grab(settings.MONITOR)),
                                        cv2.COLOR_BGRA2BGR)[230:, :, :]
    img = cv2.resize(img, (settings.WIDTH, settings.HEIGHT))
    img_name = 'MWH_Runway04_' + time_period + '_' + weather + '_' + str(episodeNum) + '_' + str(currStep) + '.png'
    # For now, just save the image to an output directory
    cv2.imwrite('%s%s' % (outDir, img_name), img)

    # Append everything to the csv file
    with open(csvFile, 'a') as fd:
	    fd.write("%s,%f,%f,%f,%f,%f,%f,%f,%f,%d,%d\n" % (img_name, absolute_time, time_step,
                cte, cte / 10.0, dtp, dtp / 2982.0, he, he / 30.0, period_of_day, cloud_cover))


if __name__ == "__main__":
	main()
