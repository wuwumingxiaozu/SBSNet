import os

import cv2
from psd_tools import PSDImage

name = os.listdir('PsdImage')

for i in name:
    psd = PSDImage.open(os.path.join('PsdImage', i))
    for index, layer in enumerate(psd.descendants()):
        if index == 0:
            layer.composite().save(os.path.join('PNGImages', i.replace('psd', 'png')))
        if index == 2:
            layer.composite().save(os.path.join('SegmentationClass', i.replace('psd', 'png')))

            image = cv2.imread(os.path.join('SegmentationClass', i.replace('psd', 'png')), 0)
            image[image == 255] = 0
            image[image > 0] = 1

            cv2.imencode('.png', image)[1].tofile(os.path.join('SegmentationClass', i))
    print(i)
