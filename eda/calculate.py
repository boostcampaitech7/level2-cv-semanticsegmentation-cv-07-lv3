import numpy as np
import cv2

def calculate_brightness(image, points):
    # 1. 이미지 읽기
    
    if image is None:
        raise ValueError("Image not found or unable to load!")
    
    # 2. 다각형 마스크 생성
    mask = np.zeros_like(image, dtype=np.uint8) 
    points = np.array(points, dtype=np.int32)    
    cv2.fillPoly(mask, [points], 255)            

    # 3. 다각형 내부 픽셀 추출
    masked_pixels = cv2.bitwise_and(image, image, mask=mask)

    # 4. 밝기 계산
    brightness = cv2.mean(masked_pixels, mask=mask)[0]
    return brightness

def calculate_polygon_area(points):
    points = np.array(points)
    x = points[:, 0]
    y = points[:, 1]
    
    area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
    return area