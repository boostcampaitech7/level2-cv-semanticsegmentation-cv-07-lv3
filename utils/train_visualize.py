import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import random
from utils_for_visualizer import reformat_metadata

os.chdir(os.path.dirname(os.path.abspath(__file__)))

def visualize_and_save(dcm_folder, json_folder, output_base_folder, metadata_file, is_labeled=False):
    
    metadata = pd.read_csv(metadata_file)

    for id_folder in os.listdir(dcm_folder):
        image_dir = os.path.join(dcm_folder, id_folder)
        json_dir = os.path.join(json_folder, id_folder)
        output_dir = os.path.join(output_base_folder)

        if os.path.isdir(image_dir) and os.path.isdir(json_dir):
            os.makedirs(output_dir, exist_ok=True)

            for image_file in os.listdir(image_dir):
                if image_file.endswith('.png'):
                    image_path = os.path.join(image_dir, image_file)
                    json_file = image_file.replace('.png', '.json')
                    json_path = os.path.join(json_dir, json_file)

                    if os.path.exists(json_path):
                        # 파일 이름에 폴더 이름 추가
                        new_image_name = f"{id_folder}_{image_file}"
                        output_path = os.path.join(output_dir, new_image_name)

                        image = Image.open(image_path)
                        fig, ax = plt.subplots(1, figsize=(12, 12))
                        ax.imshow(image)

                        with open(json_path, 'r') as f:
                            data = json.load(f)

                        # 클래스별 개수 세기
                        class_count = {}
                        annotation_count = 0

                        for item in data['annotations']:
                            class_name = item['label']
                            class_count[class_name] = class_count.get(class_name, 0) + 1
                            annotation_count += 1
                        

                        # 각 클래스의 폴리곤 마스크와 클래스 이름 출력
                        for item in data['annotations']:
                            points = item['points']
                            class_name = item['label']
                            color = [random.random() for _ in range(3)]
                            polygon = patches.Polygon(points, closed=True, fill=True, edgecolor=None, facecolor=color, alpha=0.3)
                            ax.add_patch(polygon)

                            if is_labeled:
                                # 클래스 이름 텍스트로 표시 - 마스크 영역 옆에 배치
                                centroid_x = sum([p[0] for p in points]) / len(points)
                                centroid_y = sum([p[1] for p in points]) / len(points)

                                offset_x = 10  # 마스크 오른쪽으로 약간 비껴 배치
                                ax.text(centroid_x + offset_x, centroid_y, class_name, color='white', fontsize=10, ha='left', va='center')

                        # 메타데이터 가져오기
                        image_id = id_folder[2:].strip().zfill(3)
                        metadata_row = metadata[metadata['ID'].astype(str).str.zfill(3) == image_id]

                        if not metadata_row.empty:
                            metadata_text = reformat_metadata(metadata_row, class_count, annotation_count)
                            ax.text(0.01, 0.99, metadata_text, transform=ax.transAxes, fontsize=12, color='white',
                                    ha='left', va='top')

                        plt.axis('off')
                        plt.savefig(output_path, bbox_inches='tight', pad_inches=0, dpi=300)
                        plt.close()
                        print(f"{new_image_name} 저장 완료")


def main():
    dcm_folder = "../../data/train/DCM"
    json_folder = "../../data/train/outputs_json"
    metadata_file = "../../data/meta_data.csv"

    visualize_and_save_output_folder = "../../img/train_visualized"
    #                                                                                           ▼ False일 때 레이블 출력 x
    visualize_and_save(dcm_folder, json_folder, visualize_and_save_output_folder, metadata_file, True)


if __name__ == "__main__":
    main()
