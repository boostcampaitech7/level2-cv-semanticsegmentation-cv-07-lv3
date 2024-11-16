import os
import cv2
import pandas as pd
import numpy as np
from tqdm import tqdm
from resources import PALETTE


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
    
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(csv_path)
    

    for image_name in tqdm(df['image_name'].unique()):
        
        full_image_path = find_image_in_subfolders(image_path, image_name)
        
        if full_image_path is None:
            print(f"Image not found: {os.path.join(image_path, image_name)}")
            continue
            
        image = cv2.imread(full_image_path)
        if image is None:
            print(f"Failed to read image: {full_image_path}")
            continue
            
        masks = []
        
        # 해당 이미지의 모든 클래스에 대해 마스크 생성
        image_df = df[df['image_name'] == image_name]
        for _, row in image_df.iterrows():
            if pd.isna(row['rle']):
                continue
            class_mask = rle_decode(row['rle'], image.shape[:2])
            masks.append(class_mask)
        

        colored_mask = np.zeros_like(image)
        for i, mask in enumerate(masks):
            
            color = PALETTE[i % len(PALETTE)]
            colored_mask[mask == 1] = color
        
        result = cv2.addWeighted(image, 0.7, colored_mask, 0.3, 0)
        
        
        output_path = os.path.join(output_dir, image_name)
        cv2.imwrite(output_path, result)


def main():
    """
        [check_point_csv]
            - 시각화하고자 하는 inference 결과 파일
            - code파일 내 csv파일 경로를 가져온다
            - 예시 : efficient_unet_best_model.csv
    """
    check_point_csv = 'efficient_unet_best_model.csv'

    image_path = "../../data/test/DCM"
    csv_path = f"../../code/{check_point_csv}"
    output_dir = f"../../img/test_visualized/{check_point_csv.split('_best_model')[0]}"
    
    visualize_segmentation(image_path, csv_path, output_dir)


if __name__ == "__main__":
    main()