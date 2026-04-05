import numpy as np
import cv2
import sys

img = cv2.imread("red_ball.jpg")
if img is None:
    sys.exit("Could not read the image")

def imshow(title, image):
    cv2.imshow(title, image)
    k = cv2.waitKey(0)
    cv2.destroyAllWindows()
    return

imshow("Original", img)

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
imshow("HSV", hsv)

lower_bound = np.array([0,100,50])
upper_bound = np.array([10,255,255])
lower_red2 = np.array([170, 120, 50])
upper_red2 = np.array([180, 255, 255])

mask = cv2.inRange(hsv,lower_bound,upper_bound)
mask2 = cv2.inRange(hsv, lower_red2, upper_red2)

red_mask = cv2.bitwise_or(mask, mask2)

imshow('Mask red', red_mask)

kernel = np.ones((12,12), np.uint8)

binr = cv2.threshold(red_mask,0,255,cv2.THRESH_OTSU)[1]

closing = cv2.morphologyEx(binr,cv2.MORPH_CLOSE,kernel,iterations=1)
imshow("Closing", closing)

final_mask = cv2.threshold(closing, 0, 255, cv2.THRESH_OTSU)[1]

M = cv2.moments(final_mask)
print(M)

cX = int(M["m10"] / M["m00"])
cY = int(M["m01"] / M["m00"])

cv2.circle(img,(cX,cY),5,(255,255,255),-1)
cv2.putText(img,"Czerwona pilka",(cX-25,cY-25),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),2)
imshow("Red ball with centroid" , img)







