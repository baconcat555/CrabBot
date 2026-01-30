from picamera2 import Picamera2
import cv2
import numpy as np
import math

# 1. Load an image or use a webcam feed
# For static image:
img = cv2.imread('image.jpg')
# For webcam:
# cap = cv2.VideoCapture(0)
# ret, img = cap.read()

# 2. Convert the image from BGR to HSV color space
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# 3. Define the lower and upper bounds for the color you want to detect (e.g., green)
# In OpenCV, Hue (H) range is [0, 179], Saturation (S) range is [0, 255], Value (V) range is [0, 255]
# Example values for a common green color range:
lower_green = np.array([36, 25, 25])
upper_green = np.array([86, 255, 255])

# 4. Create a binary mask using cv2.inRange()
# Pixels within the range become white (255), otherwise black (0)
mask = cv2.inRange(hsv, lower_green, upper_green)

# 5. Apply the mask to the original image using a bitwise AND operation
result = cv2.bitwise_and(img, img, mask=mask)

# 6. Display the original image, the mask, and the result
cv2.imshow('Original Image', img)
cv2.imshow('Mask', mask)
cv2.imshow('Detected Color Result', result)
cv2.waitKey(0)
cv2.destroyAllWindows()

# For a complete code example with trackbars for live H/S/V tuning, refer to the [OpenCV documentation](