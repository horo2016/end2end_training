import cv2
import numpy as np
# center定义
y_bias = 190
gray_min = np.array([0, 0, 46])
gray_max = np.array([180, 43, 255])



image_name = '../dataset/20230522154712439309.jpg'
srcimg = cv2.imread(image_name)
#srcimg = cv2.resize(srcimg, (512, 256))

cv2.imshow("srcimg",srcimg)

hsv = cv2.cvtColor(srcimg, cv2.COLOR_BGR2HSV)
mask_gray = cv2.inRange(hsv, gray_min, gray_max)
print(mask_gray.shape)
#gray = cv2.cvtColor(srcimg, cv2.COLOR_BGR2GRAY)
#cv2.imshow("gray",mask_gray)
# 大津法二值化
retval, dst = cv2.threshold(mask_gray, 30, 255, cv2.THRESH_BINARY)
# 膨胀，白区域变大
dst = cv2.dilate(dst, None, iterations=2)
# # 腐蚀，白区域变小
# dst = cv2.erode(dst, None, iterations=6)
#矩阵切片，把需要的东西提取出来
hawk = dst[180:250,18:238]

#c = cv.bitwise_and(a, b, mask=mask)  # 添加掩膜
#d = cv2.bitwise_not(hawk)  # 不加掩膜
cv2.imshow("hawk.jpg",hawk)
#cv2.imshow("dst",dst)
cv2.waitKey(0) 