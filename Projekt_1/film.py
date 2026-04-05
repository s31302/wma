import cv2
import numpy as np
import sys
#demo wersja
cap = cv2.VideoCapture('rgb_ball_720.mp4')

while cap.isOpened():
    ret, frame = cap.read()
    # Check if frame is read correctly; ret is True if successful
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    # Convert frame to grayscale
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    #ogarnij te zakresy jak to dziala
    lower_bound = np.array([0, 180, 125])
    upper_bound = np.array([10, 255, 255])
    lower_red2 = np.array([170, 170, 120])
    upper_red2 = np.array([180, 255, 255])

    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)

    red_mask = cv2.bitwise_or(mask, mask2)

    #po co to?
    binr = cv2.threshold(red_mask, 0, 255, cv2.THRESH_OTSU)[1]

    # define the kernel
    kernel = np.ones((10, 10), np.uint8)
    # opening the image
    opening = cv2.morphologyEx(binr, cv2.MORPH_OPEN, kernel, iterations=1)

    kernel = np.ones((12, 12), np.uint8)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=1)

    #po co to?
    ret, thresh = cv2.threshold(closing, 127, 255, 0)

    segmented_img = cv2.bitwise_and(frame, frame, mask=mask)

    contours, hierarchy = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    temp_output = cv2.drawContours(segmented_img, contours, -1, (0, 0, 255), 3)

    output = cv2.drawContours(frame, contours, -1, (0, 0, 255), 3)

    M = cv2.moments(thresh)

    # Sprawdzamy, czy pole powierzchni (m00) jest większe od zera
    if M["m00"] > 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])

        # Rysujemy tylko wtedy, gdy coś znaleźliśmy
        cv2.circle(frame, (cX, cY), 5, (255, 255, 255), -1)
        cv2.putText(frame, "Czerwona pilka", (cX - 25, cY - 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    else:
        print("Nie wykryto czerwonego koloru w tej klatce.")

    # Wyświetl oryginał z narysowanym kółkiem, a nie HSV (lepiej widać efekt)
    cv2.imshow('frame, click q to quit', frame)
    if cv2.waitKey(1) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
