import os
import json
import cv2
import numpy as np
import pandas as pd
from utils_for_visualizer import reformat_metadata

def visualize_and_save(dcm_folder, json_folder, output_base_folder, metadata_file, is_labeled=False, id_list=None):
    """
    GT에 대해 폴리곤을 그리는 함수
    validation 시각화에 재사용하기 위해 원래 해상도와 포맷을 유지한다
    mask 경계 + 메타데이터 + 레이블이 출력된다
    """
    metadata = pd.read_csv(metadata_file)

    for id_folder in os.listdir(dcm_folder):

        if id_list and id_folder not in id_list:
            continue

        image_dir = os.path.join(dcm_folder, id_folder)
        json_dir = os.path.join(json_folder, id_folder)
        output_dir = os.path.join(output_base_folder)

        if os.path.isdir(image_dir) and os.path.isdir(json_dir):
            os.makedirs(output_dir, exist_ok=True)

            for image_file in os.listdir(image_dir):
                if image_file.endswith('.png') or image_file.endswith('.jpg'):
                    image_path = os.path.join(image_dir, image_file)
                    json_file = image_file.replace('.png', '.json').replace('.jpg', '.json')
                    json_path = os.path.join(json_dir, json_file)

                    if os.path.exists(json_path):
                        new_image_name = f"{id_folder}_{image_file}"
                        output_path = os.path.join(output_dir, new_image_name)

                        image = cv2.imread(image_path)
                        if image is None:
                            print(f"Failed to read image: {image_path}")
                            continue


                        with open(json_path, 'r') as f:
                            data = json.load(f)

                        for item in data['annotations']:
                            points = np.array(item['points'], dtype=np.int32)
                            class_name = item['label']

                            # 폴리곤 경계 그리기
                            cv2.polylines(image, [points], isClosed=True, color=(0, 255, 0), thickness=2)

                            if is_labeled:
                                # 클래스 이름 표시
                                centroid = np.mean(points, axis=0).astype(int)
                                cv2.putText(
                                    image, class_name, (centroid[0] + 10, centroid[1] + 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 1, cv2.LINE_AA
                                )

                        # 메타데이터 가져오기
                        image_id = id_folder[2:].strip().zfill(3)
                        metadata_row = metadata[metadata['ID'].astype(str).str.zfill(3) == image_id]

                        if not metadata_row.empty:
                            metadata_text = reformat_metadata(metadata_row)

                            y_offset = 50
                            for i, line in enumerate(metadata_text.split('\n')):
                                y = 30 + i * y_offset
                                cv2.putText(
                                    image, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA
                                )


                        cv2.imwrite(output_path, image)
                        print(f"{new_image_name} 저장 완료")


def main():
    dcm_folder = "../../data/train/DCM"
    json_folder = "../../data/train/outputs_json"
    metadata_file = "../../data/meta_data.csv"

    visualize_and_save_output_folder = "../../img/train_visualized"


    # 시각화할 ID 리스트 (예: ID000, ID001 등)
    # 비어있을 경우 전체 ID에 대해 시각화를 수행
    target_ids = [
        #'ID058',
        ]

    visualize_and_save(dcm_folder, json_folder, visualize_and_save_output_folder, metadata_file, 
                       # ▼ False일 때 레이블 출력 x
                       True, id_list=target_ids)


if __name__ == "__main__":
    main()
