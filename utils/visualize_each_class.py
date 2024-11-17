import os
import json
import cv2
import math
import numpy as np
from resources import CLASSES


def visualize_one_class_with_annotations(dcm_folder, json_folder, output_base_folder, class_filter=None, batch_size=9):
    """
    이미지와 JSON GT를 읽어와 특정 클래스에 대해 마스크를 시각화하고 batch_size장씩 그리드로 저장.

    Args:
        dcm_folder (str): 이미지 파일이 저장된 폴더 경로.
        json_folder (str): JSON 파일이 저장된 폴더 경로.
        output_base_folder (str): 결과 이미지 저장 경로.
        class_filter (str): 처리할 특정 클래스 이름 (기본값: None).
        batch_size (int): 한 번에 처리할 이미지 개수 (기본값: 9).
    """

    image_files = []

    for id_folder in os.listdir(dcm_folder):
        image_dir = os.path.join(dcm_folder, id_folder)
        json_dir = os.path.join(json_folder, id_folder)

        if os.path.isdir(image_dir) and os.path.isdir(json_dir):
            for image_file in os.listdir(image_dir):
                if image_file.endswith(".png"):
                    image_path = os.path.join(image_dir, image_file)
                    json_path = os.path.join(
                        json_dir, image_file.replace(".png", ".json")
                    )
                    if os.path.exists(json_path):
                        image_files.append((image_path, json_path, id_folder))

    
    batches = [
        image_files[i : i + batch_size] for i in range(0, len(image_files), batch_size)
    ]

    for batch_idx, batch in enumerate(batches):
        processed_images = []

        for image_path, json_path, id_folder in batch:
            
            # 이미지, json, mask 준비
            img = cv2.imread(image_path)

            with open(json_path, "r") as f:
                data = json.load(f)

            mask = np.zeros(img.shape[:2], dtype=np.uint8)
            
            for annotation in data["annotations"]:
                if class_filter is None or annotation["label"] == class_filter:
                    points = np.array(annotation["points"], dtype=np.int32)
                    cv2.fillPoly(mask, [points], color=255)

            img_with_mask = cv2.addWeighted(img, 0.7, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), 0.3, 0)

            # 이미지 좌상단에 ID 출력
            cv2.putText(img_with_mask, f"{id_folder}", (40, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 3, cv2.LINE_AA,)

            processed_images.append(img_with_mask)

        grid_size = calculate_grid_size(batch_size)
        grid_image = create_image_grid(processed_images, grid_size)

        batch_output_path = os.path.join(
            output_base_folder, f"batch_{batch_idx + 1}.png"
        )
        os.makedirs(output_base_folder, exist_ok=True)
        cv2.imwrite(batch_output_path, grid_image)
        print(f"Batch {batch_idx + 1} saved at {batch_output_path}")


def calculate_grid_size(batch_size):
    cols = math.ceil(math.sqrt(batch_size))  # 열의 개수
    rows = math.ceil(batch_size / cols)     # 행의 개수
    return (rows, cols)


def create_image_grid(images, grid_size=(3, 3), base_size=900):
    """
    여러 이미지를 그리드 형태로 합침.

    Args:
        images (list): 이미지 리스트.
        grid_size (tuple): 그리드 크기 (행, 열).
        image_size (tuple): 각 이미지 크기 (기본값: 300x300).
    
    Returns:
        np.ndarray: 그리드로 합쳐진 이미지.
    """
    grid_h, grid_w = grid_size
    img_h, img_w = int(base_size/grid_h), int(base_size/grid_w)

    grid_canvas = np.zeros((grid_h * img_h, grid_w * img_w, 3), dtype=np.uint8)

    for idx, img in enumerate(images):
        if idx >= grid_h * grid_w:
            break

        resized_img = cv2.resize(img, (img_w, img_h))
        
        row = idx // grid_w
        col = idx % grid_w

        start_y, start_x = row * img_h, col * img_w
        grid_canvas[start_y:start_y + img_h, start_x:start_x + img_w] = resized_img

    return grid_canvas


def main():
    data_root = '../../data/train'

    dcm_folder = data_root + "/DCM"  # 입력 폴더 경로
    json_folder = data_root + "/outputs_json" 

    # 0 : 'finger-1', 1 : 'finger-2', 2 : 'finger-3', 3 : 'finger-4', 4 : 'finger-5', 
    # 5 : 'finger-6', 6 : 'finger-7', 7 : 'finger-8', 8 : 'finger-9', 9 : 'finger-10', 
    # 10 : 'finger-11', 11 : 'finger-12', 12 : 'finger-13', 13 : 'finger-14', 14 : 'finger-15', 
    # 15 : 'finger-16', 16 : 'finger-17', 17 : 'finger-18', 18 : 'finger-19', 19 : 'Trapezium', 
    # 20 : 'Trapezoid', 21 : 'Capitate', 22 : 'Hamate', 23 : 'Scaphoid', 24 : 'Lunate', 
    # 25 : 'Triquetrum', 26 : 'Pisiform', 27 : 'Radius', 28 : 'Ulna'

    class_filter = CLASSES[26]
    batch_size = 9

    output_base_folder = f"../../img/visualization_of_{class_filter}"  # 출력 폴더 경로
    os.makedirs(output_base_folder, exist_ok=True)
    
    visualize_one_class_with_annotations(dcm_folder, json_folder, output_base_folder, class_filter, batch_size)


if __name__ == "__main__":
    main()