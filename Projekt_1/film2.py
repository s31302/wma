#KOD NAPISANY PRZEZ CHAT

import cv2
import numpy as np

cap = cv2.VideoCapture('rgb_ball_720.mp4')

# Definiujemy kernele przed pętlą (oszczędność procesora)
kernel_open = np.ones((5, 5), np.uint8)
kernel_close = np.ones((15, 15), np.uint8)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Zakresy (podniesione nasycenie S, by odciąć bladość skóry)
    lower_red1 = np.array([0, 160, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 160, 100])
    upper_red2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(mask1, mask2)

    # MORFOLOGIA: najpierw OPEN (usuwa szum/rękę), potem CLOSE (łata dziury w piłce)
    mask_cleaned = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel_open)
    mask_cleaned = cv2.morphologyEx(mask_cleaned, cv2.MORPH_CLOSE, kernel_close)

    # Szukamy konturów na WYCZYSZCZONEJ masce
    contours, _ = cv2.findContours(mask_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # KLUCZ: Wybieramy tylko największy kontur (piłkę)
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)

        # Tylko jeśli obiekt jest wystarczająco duży (np. 100 pikseli)
        if area > 100:
            # Liczymy momenty TYLKO dla tego jednego konturu
            M = cv2.moments(largest_contour)

            if M["m00"] > 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])

                # Rysujemy wynik na oryginalnym kadrze
                cv2.circle(frame, (cX, cY), 7, (0, 255, 0), -1)
                cv2.putText(frame, "Pilka", (cX - 20, cY - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Opcjonalnie: obrysuj wykrytą piłkę
                cv2.drawContours(frame, [largest_contour], -1, (0, 255, 0), 2)

    cv2.imshow('Tracking Pilki', frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()