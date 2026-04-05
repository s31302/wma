# Capture video
import numpy as np
import sys
import cv2

cap = cv2.VideoCapture('rgb_ball_720.mp4')
while cap.isOpened():
    ret, frame = cap.read()
    # Check if frame is read correctly; ret is True if successful
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # H-barwa,S-nasycenie,V-jasnosc
    lower_red1 = np.array([0, 170, 120])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 170, 110])
    upper_red2 = np.array([180, 255, 255])

    mask = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)

    red_mask = cv2.bitwise_or(mask, mask2)

    binr = cv2.threshold(red_mask, 0, 255, cv2.THRESH_OTSU)[1]

    kernel = np.ones((5, 5), np.uint8)
    opening = cv2.morphologyEx(binr, cv2.MORPH_OPEN, kernel, iterations=1)

    kernel = np.ones((25, 25), np.uint8)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=1)

    final_mask = cv2.threshold(closing, 0, 255, cv2.THRESH_OTSU)[1]

    M = cv2.moments(final_mask)
    if M["m00"] > 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])

        cv2.circle(frame, (cX, cY), 3, (255, 255, 255), -1)
        cv2.putText(frame, "Czerwona pilka", (cX - 25, cY - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Display frame; press 'q' to quit
    cv2.imshow('frame, click q to quit', frame)
    if cv2.waitKey(1) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()