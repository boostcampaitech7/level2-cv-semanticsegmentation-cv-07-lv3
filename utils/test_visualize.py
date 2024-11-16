import os
import cv2
import pandas as pd
import numpy as np
from tqdm import tqdm

# 시각화를 위한 팔레트를 설정합니다.
PALETTE = [
    (220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230), (106, 0, 228),
    (0, 60, 100), (0, 80, 100), (0, 0, 70), (0, 0, 192), (250, 170, 30),
    (100, 170, 30), (220, 220, 0), (175, 116, 175), (250, 0, 30), (165, 42, 42),
    (255, 77, 255), (0, 226, 252), (182, 182, 255), (0, 82, 0), (120, 166, 157),
    (110, 76, 0), (174, 57, 255), (199, 100, 0), (72, 0, 118), (255, 179, 240),
    (0, 125, 92), (209, 0, 151), (188, 208, 182), (0, 220, 176),
]

def find_image_in_subfolders(base_path, image_name):
    """하위 폴더들을 검색하여 이미지 파일의 전체 경로를 찾습니다."""
    for root, dirs, files in os.walk(base_path):
        if image_name in files:
            return os.path.join(root, image_name)
    return None

def rle_decode(mask_rle, shape):
    """RLE 인코딩된 마스크를 디코딩합니다."""
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape)

def visualize_segmentation(image_path, csv_path, output_dir):
    """세그멘테이션 결과를 시각화합니다."""
    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    
    # CSV 파일 읽기
    df = pd.read_csv(csv_path)
    
    # 이미지별로 처리
    for image_name in tqdm(df['image_name'].unique()):
        # 이미지 찾기
        full_image_path = find_image_in_subfolders(image_path, image_name)
        
        if full_image_path is None:
            print(f"Image not found: {os.path.join(image_path, image_name)}")
            continue
            
        # 이미지 읽기
        image = cv2.imread(full_image_path)
        if image is None:
            print(f"Failed to read image: {full_image_path}")
            continue
            
        # 마스크 초기화 (클래스별로 별도의 마스크 생성)
        masks = []
        
        # 해당 이미지의 모든 클래스에 대해 마스크 생성
        image_df = df[df['image_name'] == image_name]
        for _, row in image_df.iterrows():
            if pd.isna(row['rle']):
                continue
            class_mask = rle_decode(row['rle'], image.shape[:2])
            masks.append(class_mask)
        
        # 마스크 시각화
        colored_mask = np.zeros_like(image)
        for i, mask in enumerate(masks):
            # 각 클래스별로 다른 색상 적용
            color = PALETTE[i % len(PALETTE)]  # PALETTE를 순환하여 사용
            colored_mask[mask == 1] = color
        
        # 이미지와 마스크 합성
        result = cv2.addWeighted(image, 0.7, colored_mask, 0.3, 0)
        
        # 결과 저장
        output_path = os.path.join(output_dir, image_name)
        cv2.imwrite(output_path, result)

if __name__ == "__main__":
    image_path = "/data/ephemeral/home/data/test/DCM"
    csv_path = "/data/ephemeral/home/code/efficientb0_UnetPlusPlus_best_model.csv"
    output_dir = "/data/ephemeral/home/img/efficientb0_UnetPlusPlus_fold0"
    
    visualize_segmentation(image_path, csv_path, output_dir)