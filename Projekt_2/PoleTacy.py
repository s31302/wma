import cv2 as cv
import numpy as np


def imshow(title, image):
    cv.imshow(title,image)
    k = cv.waitKey(0)#0 infinity waiting time
    if k == ord("s"):
        cv.imwrite(f"00_{title}_saved.jpg", image)
    cv.destroyAllWindows()
    return

#im= cv.imread('tray8.jpg')
im= cv.imread('tray3.jpg')
#im= cv.imread('tray7.jpg')
assert im is not None, "file could not be read, check with os.path.exists()"
imgray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)

_, thresh = cv.threshold(imgray, 127, 255, cv.THRESH_BINARY)

im_hsv = cv.cvtColor(im, cv.COLOR_BGR2HSV)
lower_orange = np.array([7, 160, 180])
upper_orange = np.array([25, 255, 255])
mask = cv.inRange(im_hsv, lower_orange, upper_orange)

masked_image = cv.bitwise_and(im, im, mask=mask)

contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
cv.drawContours(im, contours, -1, (255,0, 255), 3)

total_area = sum(cv.contourArea(c) for c in contours)
print("Pole powierzchni tacki: ", total_area)