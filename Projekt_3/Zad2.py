import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

# Program 2: zaimportuj plik photo_2_query.jpg i photo_2_train.jpg, wykonaj i wyświetl dopasowanie cech z homografią (4 punkty).

def imshow(title, image):
    cv.imshow(title,image)
    k = cv.waitKey(0)#0 infinity waiting time
    if k == ord("s"):
        cv.imwrite(f"00_{title}_saved.jpg", image)
    cv.destroyAllWindows()
    return

MIN_MATCH_COUNT = 10

img1 = cv.imread('photo_2_train.jpg',cv.IMREAD_GRAYSCALE)
img2 = cv.imread('photo_2_query.jpg',cv.IMREAD_GRAYSCALE)

sift = cv.SIFT_create()

# Znalezienie punktow kluczowych i dyskryptorow z sift
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)

FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 50)
flann = cv.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1,des2,k=2)

#dobre cechy z testu proporcji lowea
good = []
for m, n in matches:
    if m.distance < 0.7 * n.distance:
        good.append(m)

#jesli ilosc dobrych cech sie zgadza, wydrebniamy lokalizacje pk z obu obrazkow
if len(good) > MIN_MATCH_COUNT:
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    
    # findhomography szuka transformacji perpektwicznej
    M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
    matchesMask = mask.ravel().tolist()

    h, w = img1.shape
    pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
    dst = cv.perspectiveTransform(pts, M)

    img2 = cv.polylines(img2, [np.int32(dst)], True, 255, 3, cv.LINE_AA)
else:
    print("Nie znaleziono wystarczającej liczby dopasowan- {}/{}".format(len(good), MIN_MATCH_COUNT))
    matchesMask = None


#Rysownie linerow - prawidłowe wewnętrzne dopasowania (jeśli udało się znaleźć obiekt) lub pasujące punkty kluczowe (jeśli się nie udało)
draw_params = dict(matchColor = (0,255,0), # dopasowanie narysowane na zielono
                   singlePointColor = None,
                   matchesMask = matchesMask, # narusuje tylko linery
                   flags = 2)
img3 = cv.drawMatches(img1, kp1, img2, kp2, good, None, **draw_params)
plt.imshow(img3, 'gray'), plt.show()
cv.imwrite("Dopasowane cechy.jpg", img3)