from picamera2 import Picamera2
import cv2
import numpy as np
import math

from gpiozero import LED


led1 = LED(2) #GPIO 5
led2 = LED(3) # GPIO 6
led3 = LED(4)
led4 = LED(17)
led5 = LED(27)

picam2 = Picamera2()
picam2.configure(
    picam2.create_preview_configuration(
        main={"size": (640, 480), "format": "RGB888"}
    )
)
picam2.start()

while True:
    img = picam2.capture_array()

    # Define ROI for hand
    cv2.rectangle(img, (80, 80), (400, 400), (0, 255, 0), 2)
    crop_img = img[110:400, 110:400]

    # Convert to grayscale
    grey = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)

    # Blur
    blurred = cv2.GaussianBlur(grey, (5, 5), 0)

    # Threshold
    _, thresh1 = cv2.threshold(
        blurred, 0, 150,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    cv2.imshow('Thresholded', thresh1)

    # Find contours
    contours, hierarchy = cv2.findContours(
        thresh1.copy(),
        cv2.RETR_TREE,
        cv2.CHAIN_APPROX_NONE
    )

    if len(contours) == 0:
        cv2.imshow('Gesture', img)
        if cv2.waitKey(10) == 27:
            break
        continue

    # Max contour
    cnt = max(contours, key=cv2.contourArea)

    # Bounding box
    x, y, w, h = cv2.boundingRect(cnt)
    cv2.rectangle(crop_img, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # Convex hull
    hull = cv2.convexHull(cnt)
    drawing = np.zeros(crop_img.shape, np.uint8)
    cv2.drawContours(drawing, [cnt], 0, (0, 255, 0), 2)
    cv2.drawContours(drawing, [hull], 0, (0, 0, 255), 2)

    # Convexity defects
    hull_indices = cv2.convexHull(cnt, returnPoints=False)
    defects = cv2.convexityDefects(cnt, hull_indices)

    count_defects = 0

    if defects is not None:
        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]
            start = tuple(cnt[s][0])
            end = tuple(cnt[e][0])
            far = tuple(cnt[f][0])

            a = math.dist(start, end)
            b = math.dist(start, far)
            c = math.dist(end, far)

            angle = math.acos((b*b + c*c - a*a) / (2*b*c)) * 57

            if angle <= 90:
                count_defects += 1
                cv2.circle(crop_img, far, 5, [0, 0, 255], -1)

            cv2.line(crop_img, start, end, [0, 255, 0], 2)

    # Finger count display
    if count_defects == 0:
        text = "1 Finger"
        led1.on()
        led2.off()
        led3.off()
        led4.off()
        led5.off()

    elif count_defects == 1:
        text = "2 Fingers"
        led1.off()
        led2.on()
        led3.off()
        led4.off()
        led5.off()
    elif count_defects == 2:
        text = "3 Fingers"
        led1.off()
        led2.off()
        led3.on()
        led4.off()
        led5.off()
    elif count_defects == 3:
        text = "4 Fingers"
        led1.off()
        led2.off()
        led3.off()
        led4.on()
        led5.off()
    else:
        text = "5 Fingers"
        led1.off()
        led2.off()
        led3.off()
        led4.off()
        led5.on()


    cv2.putText(img, text, (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow('Gesture', img)
    cv2.imshow('Contours', np.hstack((drawing, crop_img)))

    if cv2.waitKey(10) == 27:
        break

cv2.destroyAllWindows()
picam2.stop()
