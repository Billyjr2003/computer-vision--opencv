import cv2 as cv 
import numpy as np 



#read images :
img_1 = cv.imread('project2/Screen Shot 2023-02-14 at 12.52.36 AM.png')

img_2 = cv.imread('project2/Screen Shot 2023-02-14 at 12.52.56 AM.png')

#images should be at the same size
#Gray scale 
Gray_1 = cv.cvtColor(img_1,cv.COLOR_BGR2GRAY)
Gray_2 = cv.cvtColor(img_2,cv.COLOR_BGR2GRAY)

#find the difference between the two images 
diff = cv.absdiff(Gray_1,Gray_2)





#Apply thresh
thresh = cv.threshold(diff, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)[1]
process_img = cv.bitwise_not(thresh)
kernel = np.ones((5,5),np.uint8)

opening = cv.morphologyEx(process_img, cv.MORPH_OPEN, kernel)
#finding contours
contours , hierarchy= cv.findContours(opening, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
for contour in contours :
    if cv.contourArea(contour)> 100:
    
        x, y, w, h = cv.boundingRect(contour)
        cv.rectangle(img_2, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv.rectangle(img_1, (x, y), (x + w, y + h), (0, 255, 0), 2)

stack_img = np.hstack([img_1,img_2])
#show images


cv.imshow('final result',stack_img)





cv.waitKey(0)
cv.destroyAllWindows()