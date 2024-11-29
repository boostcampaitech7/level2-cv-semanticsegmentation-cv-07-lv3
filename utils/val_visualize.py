import os
import re
import cv2
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
from resources import PALETTE
from utils_for_visualizer import decode_rle_to_mask


def list_files(folder_path, extensions=None):
    if not os.path.exists(folder_path):
        print(f"Folder does not exist: {folder_path}")
        return []
    
    file_list = []
    for _, _, files in os.walk(folder_path):
        for file in files:
            if extensions is None or file.lower().endswith(extensions):
                file_list.append(file)
    
    return file_list


def visualize_validation(image_path, csv_path, output_dir, json_folder=None):
    
    os.makedirs(output_dir, exist_ok=True)
    data = pd.read_csv(csv_path)

    # 예시 : [ID052_image1661390896968.png, ..]
    file_list = list_files(image_path, ".png")


    for image_name in tqdm(data['image_name'].unique()):
            
        matching_file = next((f for f in file_list if image_name in f), None)

        if matching_file is None:
            print(f"Image not found in file list: {image_name}")
            continue

        # ID 추출 (파일명에서 ID 추출, 예: ID052)
        id_match = re.match(r'(ID\d+)_', os.path.basename(matching_file))

        if id_match:
            id_folder = id_match.group(1)
        else:
            print(f"Failed to extract ID from: {matching_file}")
            continue

        full_image_path = os.path.join(image_path, f'{id_folder}_{image_name}')

        if full_image_path is None:
            print(f"Image not found: {full_image_path}")
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
            
            height, width = image.shape[:2]
            class_mask = decode_rle_to_mask(row['rle'], height, width)
            
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

        output_path = os.path.join(output_dir, f'{id_folder}_{image_name}')
        cv2.imwrite(output_path, result)


def main():
    """
        [★★선행 작업★★]
            - gt_visualize.py를 실행하여 얻은 이미지가 img/train_visualized/ 폴더에 있어야만 사용 가능하다

        [시각화할 validation 결과 파일 설정 필요]
            - csv_path(기본 경로) : data/ephemeral/home/validation_result

            - validation_csv : 경로 내 csv 파일
                - 조건 : (image_name, class, rle) 형식이어야 한다
                - 파일 이름 예시 : val_epoch_1.csv
        
        [출력 파일이 저장되는 곳]
            - output_dir(기본 경로) : data/ephemeral/home/img/val_visualized/{validation_csv}
    """

    # validation하고 싶은 csv 파일
    validation_csv = 'val_epoch_95.csv'
    csv_path = f"../../validation_result/smp_unetplusplus_efficientb0/{validation_csv}"

    # data 폴더 경로
    img_root = '../../img'
    data_root = '../../data/train'

    json_folder = os.path.join(data_root, "outputs_json")

    image_path = os.path.join(img_root, "train_visualized")

    # 출력 경로
    output_dir = f"{img_root}/val_visualized/{validation_csv}"

    visualize_validation(image_path, csv_path, output_dir, json_folder)
    

if __name__ == "__main__":
    main()