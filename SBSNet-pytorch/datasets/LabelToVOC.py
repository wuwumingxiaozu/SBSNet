import os

import cv2 as cv

name = os.listdir('SegmentationClass')

for i in name:
    image = cv.imread(os.path.join('SegmentationClass', i), 0)
    image[image == 255] = 0
    image[image > 0] = 1

    cv.imencode('.png', image)[1].tofile(os.path.join('SegmentationClass', i))
