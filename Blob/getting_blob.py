import numpy as np
import cv2
import time

image_BGR = cv2.imread('woman-working.png')
cv2.namedWindow('Original Image', cv2.WINDOW_NORMAL)
# Pay attention! 'cv2.imshow' takes images in BGR format
cv2.imshow('Original Image', image_BGR)

cv2.waitKey(0)

cv2.destroyWindow('Original Image')

blob = cv2.dnn.blobFromImage(image_BGR, 1 / 255.0, (416, 416),
                             swapRB=True, crop=False)

# Check point
print('Image shape:', image_BGR.shape)  # (511, 767, 3)
print('Blob shape:', blob.shape)  # (1, 3, 416, 416)

blob_to_show = blob[0, :, :, :].transpose(1, 2, 0)
print(blob_to_show.shape)  # (416, 416, 3)

cv2.namedWindow('Blob Image', cv2.WINDOW_NORMAL)
cv2.imshow('Blob Image', cv2.cvtColor(blob_to_show, cv2.COLOR_RGB2BGR))
cv2.waitKey(0)
cv2.destroyWindow('Blob Image')
