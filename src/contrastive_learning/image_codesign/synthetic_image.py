"""
    inputs: 
        - num distractors
        - discractor shape and color: ellipse
        - landing site
            - box
            - (x, y) center of box randomly sampled
            - xwidth, ywidth randomly sampled from a range
            - color randomly sampled with hue etc.
        - robot:
            - also an ellipse, gray 

    useful links:
        - https://scikit-image.org/docs/stable/auto_examples/edges/plot_random_shapes.html#sphx-glr-auto-examples-edges-plot-random-shapes-py

"""

import math
import numpy as np
import matplotlib.pyplot as plt

from skimage.draw import (line, polygon,  \
                          circle_perimeter, \
                          ellipse, ellipse_perimeter, \
                          bezier_curve)

if __name__ == '__main__':

    NUM_IMAGES = 5

    W = 224
    H = 224
    C = 3

    for i in range(NUM_IMAGES):

        img = np.zeros((W, H, C), dtype=np.double)

        # first plot the landing site


        # draw line
        rr, cc = line(120, 123, 20, 400)
        img[rr, cc, 0] = 255

        # fill polygon
        poly = np.array((
            (300, 300),
            (480, 320),
            (380, 430),
            (220, 590),
            (300, 300),
        ))
        rr, cc = polygon(poly[:, 0], poly[:, 1], img.shape)
        img[rr, cc, 1] = 1

        # fill ellipse
        rr, cc = ellipse(300, 300, 100, 200, img.shape)
        img[rr, cc, 2] = 1

        # circle
        rr, cc = circle_perimeter(120, 400, 15)
        img[rr, cc, :] = (1, 0, 0)

        # Bezier curve
        rr, cc = bezier_curve(70, 100, 10, 10, 150, 100, 1)
        img[rr, cc, :] = (1, 0, 0)

        # ellipses
        rr, cc = ellipse_perimeter(120, 400, 60, 20, orientation=math.pi / 4.)
        img[rr, cc, :] = (1, 0, 1)
        rr, cc = ellipse_perimeter(120, 400, 60, 20, orientation=-math.pi / 4.)
        img[rr, cc, :] = (0, 0, 1)
        rr, cc = ellipse_perimeter(120, 400, 60, 20, orientation=math.pi / 2.)
        img[rr, cc, :] = (1, 1, 1)

        plt.imshow(img)
        plt.axis('off')
        plt.savefig('test_' + str(i) + '.pdf')
        plt.close()


