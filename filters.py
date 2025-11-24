import cv2
import numpy
import utils

def strokeEdges(src, dst, blurKsize=7, edgeKsize=5):
    if blurKsize >= 3:
        blurredSrc = cv2.medianBlur(src, blurKsize)
        graySrc = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

    channels = cv2.split(src)
    normalizedInverseAlpha = (1.0 / 255) * (255 - graySrc)
    for channel in channels:
        channel[:] = channel * normalizedInverseAlpha
    cv2.merge(channels, dst)
