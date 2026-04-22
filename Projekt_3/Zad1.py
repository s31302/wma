import cv2 as cv
import numpy as np

#- Program 1: zaimportuj plik photo_1.jpg i: znajdź i wyświetl tylko 4 najsilniejsze narożniki za pomocą funkcji wykrywania narożników Harrisa (2 punkty),
#a następnie znajdź i wyświetl punkty kluczowe za pomocą metody SIFT (2 punkty).
def imshow(title, image):
    cv.imshow(title,image)
    k = cv.waitKey(0)#0 infinity waiting time
    if k == ord("s"):
        cv.imwrite(f"00_{title}_saved.jpg", image)
    cv.destroyAllWindows()
    return
def show_smaller(title, image, scale=0.5):
    resized = cv.resize(image, None, fx=scale, fy=scale)
    cv.imshow(title, resized)
    cv.waitKey(0)
    #cv.destroyAllWindows()


img = cv.imread('photo_1.jpg')
assert img is not None, "file could not be read, check with os.path.exists()"

img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

mask = cv.inRange(img_hsv, (0, 50, 50), (180, 255, 255))

gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
#gray_masked = cv.bitwise_and(gray, gray, mask=mask)
#gray_blur = cv.GaussianBlur(gray_masked, (3,3), 0)
#gray_blur = cv.bilateralFilter(gray, 9, 75, 75)

show_smaller("obraz na ktorym probimy", gray)

#harris
dst = cv.cornerHarris(gray,3,3,0.02)
dst = cv.dilate(dst,None)


Threshold=0.01
z = Threshold * dst.max()

#punkty ktore spelniaa z
px, py = np.where(dst > z)

wagi = dst[px, py].tolist()
punkty = list(zip(px, py))


filtered_punkty = []
filtered_wagi = []
min_dystans = 25

posortowane_wagi = np.argsort(wagi)[::-1]

for i in posortowane_wagi:
    y, x = punkty[i]

    #sprawdznie dystanus pomiedzy punkta a dodanymi
    if all((y - y2) ** 2 + (x - x2) ** 2 > min_dystans ** 2 for y2, x2 in filtered_punkty):
        filtered_punkty.append((y, x))
        filtered_wagi.append(wagi[i])


img_all = img.copy()

for y, x in filtered_punkty:
    cv.circle(img_all, (x, y), 6, (255, 0, 0), -1)

show_smaller('Wszystkie znalezione narozniki', img_all)

#sortowanie 4 najsiliejszych
indices = np.argsort(filtered_wagi)[-4:]
img_top4 = img.copy()

for i in indices:
    y, x = filtered_punkty[i]
    cv.circle(img_top4, (x, y), 10, (255, 0, 0), -1)

show_smaller('4 najlepsze narozniki', img_top4, 0.5)

#sift
sift = cv.SIFT_create()

kp, des = sift.detectAndCompute(gray, None)

img_sift = cv.drawKeypoints(gray, kp, img, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

show_smaller('sift_keypoints.jpg', img_sift)
cv.destroyAllWindows()