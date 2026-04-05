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

img = cv.medianBlur(imgray,5)
cimg = cv.cvtColor(img,cv.COLOR_GRAY2BGR)

circles = cv.HoughCircles(img,method = cv.HOUGH_GRADIENT,dp = 1,minDist = 20,param1=150,param2=30,minRadius=0,maxRadius=100)

circles = np.uint16(np.around(circles))

for i in circles[0,:]:
    cv.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
    cv.circle(cimg,(i[0],i[1]),2,(0,0,255),3)
cv.imshow('Znalezione kola',cimg)
cv.waitKey(0)
cv.destroyAllWindows()

duzeMonetyNaTacy=0
maleMonetyNaTacy=0
duzeMonetyPozaTaca=0
maleMonetyPozaTaca=0

tray_contour = max(contours, key=cv.contourArea)
imgray_bgr = cv.cvtColor(imgray, cv.COLOR_GRAY2BGR)
cv.drawContours(imgray_bgr, [tray_contour], -1, (255,0,255), 3)
imshow('Ostateczny kontur tacy', imgray_bgr)

for i in circles[0,:]:
    x, y, r = int(i[0]), int(i[1]), int(i[2])
    #print(r)

    if cv.pointPolygonTest(tray_contour, (x,y), measureDist=False) == 1:
        kolor_monety = (0,255,0)
        if r > 32:
            duzeMonetyNaTacy+=1
        else:
            maleMonetyNaTacy+=1
    else:
        kolor_monety = (255, 0, 0)
        if r > 32:
            duzeMonetyPozaTaca+=1
        else:
            maleMonetyPozaTaca+=1
    cv.circle(cimg,(x,y), r,kolor_monety,2)
    cv.circle(cimg,(x,y),2,(0,0,255),3)

print("duze monety na tacy", duzeMonetyNaTacy)
print("male monety na tacy", maleMonetyNaTacy)
print("duze monety poza taca",duzeMonetyPozaTaca)
print("male monety poza taca", maleMonetyPozaTaca)

cv.imshow('Rozroznione monety',cimg)
cv.waitKey(0)
cv.destroyAllWindows()
