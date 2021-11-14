import numpy as np
import cv2

image_BGR = cv2.imread('woman-working.png')

cv2.namedWindow('Original Image', cv2.WINDOW_NORMAL)
cv2.imshow('Original Image', image_BGR)

cv2.waitKey(0)

cv2.destroyWindow('Original Image')

print('Image shape:', image_BGR.shape)  # (900, 1200, 3)

h, w = image_BGR.shape[:2]

print('Image height={0} and width={1}'.format(h, w))  # 900 1200
