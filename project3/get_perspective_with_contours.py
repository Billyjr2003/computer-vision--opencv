import cv2 as cv
import numpy as np 
import matplotlib.pyplot as plt 
cord = []
def call_event(event, x, y, flags, param):
    global cord
    global img
    if event == cv.EVENT_LBUTTONDOWN:
        cv.circle(img, (x, y), 5, (255, 0, 0), 2)
        cord.append([x, y])

img = cv.imread("project3/resources/clean_shape (2).jpg")
cv.namedWindow("image")
cv.setMouseCallback("image",call_event)
while True:
    cv.imshow("image", img)
    key = cv.waitKey(1)
    if key == ord("Q") or key == ord("q"):
        break
    if key == 27:
        pts_1 = np.float32(cord)
        pts_2 = np.float32([[0, 0], [300, 0], [0, 300], [300, 300]])

        
        M = cv.getPerspectiveTransform(pts_1, pts_2)

        
        dst_image = cv.warpPerspective(img, M, (300, 300))
        gray_image = cv.cvtColor(dst_image, cv.COLOR_BGR2GRAY)
        
        thresh = cv.adaptiveThreshold(gray_image, 255,cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY , 135 , 2)
        process_img = cv.bitwise_not(thresh)
        kernel = np.ones((5,5),np.uint8)
        opening = cv.morphologyEx(process_img, cv.MORPH_OPEN, kernel)
        contours, hierarchy = cv.findContours(opening, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        for contour in contours :
    
           x, y, w, h = cv.boundingRect(contour)
           cv.rectangle(dst_image, (x, y), (x + w, y + h), (0, 255, 0), 5)

        
        cv.imshow("dst_image",dst_image)

cv.destroyAllWindows()