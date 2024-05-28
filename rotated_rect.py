import cv2
import numpy as np

# 이미지 생성
image = np.zeros((500, 500, 3), dtype=np.uint8)

# 사각형의 중심, 크기, 회전 각도 지정
center = (250, 250)
size = (24, 18)  # width, height
angle = 30  # 회전 각도

# RotatedRect 객체 생성
rotated_rect = ((center[0], center[1]), (size[0], size[1]), angle)

# RotatedRect에서 4개의 꼭짓점 계산
box = cv2.boxPoints(rotated_rect)
box = np.int0(box)

# 사각형 그리기
cv2.drawContours(image, [box], 0, (0, 255, 0), 2)

# 이미지 출력
cv2.imshow("Rotated Rectangle", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
