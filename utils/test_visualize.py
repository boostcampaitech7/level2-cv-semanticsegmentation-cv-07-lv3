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

def visualize_test(image_path, csv_path, output_dir, metadata_file):
    """세그멘테이션 결과를 시각화합니다."""
    
    os.makedirs(output_dir, exist_ok=True)
    data = pd.read_csv(csv_path)
    metadata = pd.read_csv(metadata_file)
    

    for image_name in tqdm(data['image_name'].unique()):
        
        full_image_path = find_image_in_subfolders(image_path, image_name)
        
        if full_image_path is None:
            print(f"Image not found: {os.path.join(image_path, image_name)}")
            continue
            
        image = cv2.imread(full_image_path)
        if image is None:
            print(f"Failed to read image: {full_image_path}")
            continue
            
        masks = []
        class_names = []
        
        # 해당 이미지의 모든 클래스에 대해 마스크 생성
        image_df = data[data['image_name'] == image_name]
        for _, row in image_df.iterrows():
            if pd.isna(row['rle']):
                continue
            class_mask = rle_decode(row['rle'], image.shape[:2])
            masks.append(class_mask)
            class_names.append(row['class'])
        

        colored_mask = np.zeros_like(image)

        for i, mask in enumerate(masks):
            
            color = PALETTE[i % len(PALETTE)]
            colored_mask[mask == 1] = color

            y_coords, x_coords = np.where(mask == 1)
            if len(y_coords) > 0 and len(x_coords) > 0:
                centroid_x = int(np.mean(x_coords))
                centroid_y = int(np.mean(y_coords))
                text_x = min(centroid_x + 10, image.shape[1] - 100)
                text_y = max(centroid_y, 30)
                cv2.putText(image, class_names[i], (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        
        result = cv2.addWeighted(image, 0.7, colored_mask, 0.3, 0)

        # 이미지 folder 이름 추출
        id_folder = os.path.basename(os.path.dirname(full_image_path))
        metadata_row = metadata[metadata['ID'].astype(str).str.zfill(3) == id_folder[2:].zfill(3)]

        if metadata_row.empty:
            print(f"Metadata not found for image: {id_folder}/{image_name}")
            continue

        metadata_row = metadata_row.fillna("N/A")  # NaN 값을 "N/A"로 대체
        gender_kor = metadata_row['성별'].values[0]

        if '여' in gender_kor:
            gender_eng = 'female'
        elif '남' in gender_kor:
            gender_eng = 'male'
        else:
            gender_eng = 'unknown'
            

        metadata_text = (
            f"ID: {metadata_row['ID'].values[0]} \n"
            f"age: {metadata_row['나이'].values[0]}\n"
            f"gender: {gender_eng}\n"
            f"weight: {metadata_row['체중(몸무게)'].values[0]}\n"
            f"height: {metadata_row['키(신장)'].values[0]}\n"
            f"issue: {metadata_row['Unnamed: 5'].values[0]}\n"
        )

        y0, dy = 50, 50
        for i, line in enumerate(metadata_text.split('\n')):
            y = y0 + i * dy
            cv2.putText(result, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)

        new_image_name = f"{id_folder}_{image_name}"
        output_path = os.path.join(output_dir, new_image_name)
        cv2.imwrite(output_path, result)


def main():
    """
        [is_inference]
            - inference를 시각화할 때 True
            - validation을 시각화할 때 False

        [inference_csv]
            - 시각화하고자 하는 inference 결과 파일
            - code파일 내 csv파일 경로를 가져온다
            - 예시 : efficient_unet_best_model.csv
        [validation_csv]
            - 시각화할 validation 결과 파일
            - validation_result 폴더 내 csv 파일을 가져온다
            - 예시 : val_epoch_1.csv
    """
    is_inference = False

    metadata_csv = "../../data/meta_data.csv"

    if is_inference:
        inference_csv = 'efficient_unet_best_model.csv'

        image_path = "../../data/test/DCM"
        csv_path = f"../../code/{inference_csv}"
        output_dir = f"../../img/test_visualized/{inference_csv.split('_best_model')[0]}"
    else:
        validation_csv = 'val_epoch_ 1.csv'

        image_path = "../../data/train/DCM"
        csv_path = f"../../validation_result/{validation_csv}"
        output_dir = f"../../img/val_visualized/{validation_csv}"

    visualize_test(image_path, csv_path, output_dir, metadata_csv)
    
    
    


if __name__ == "__main__":
    main()