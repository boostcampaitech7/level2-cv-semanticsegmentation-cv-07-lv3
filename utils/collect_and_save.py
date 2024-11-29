import os
import shutil

def collect_and_save_images(train_folder, test_folder, output_folder):
    
    """
        train과 test에 대한 이미지를 모두 한 폴더로 저장해주는 함수
    """

    train_output_folder = os.path.join(output_folder, 'train')
    test_output_folder = os.path.join(output_folder, 'test')
    os.makedirs(train_output_folder, exist_ok=True)
    os.makedirs(test_output_folder, exist_ok=True)

    # Train 및 Test 폴더의 이미지 파일 처리
    for folder_name in ['train', 'test']:
        current_folder = os.path.join(train_folder if folder_name == 'train' else test_folder, 'DCM')
        current_output_folder = train_output_folder if folder_name == 'train' else test_output_folder

        if os.path.exists(current_folder):
            for id_folder in os.listdir(current_folder):
                folder_path = os.path.join(current_folder, id_folder)
                if os.path.isdir(folder_path):
                    for image_file in os.listdir(folder_path):
                        
                        if image_file.endswith('.png'):
                            # 파일 이름 앞에 폴더 이름(id_folder) 추가
                            new_image_name = f"{id_folder}_{image_file}"
                            output_path = os.path.join(current_output_folder, new_image_name)
                            image_path = os.path.join(folder_path, image_file)
                            
                            shutil.copy(image_path, output_path)
                            print(f"{new_image_name} 저장 완료")


def main():
    train_folder = "../../data/train"
    test_folder = "../../data/test"
    output_folder = "../../img/save_all_in_one"


    collect_and_save_images(train_folder, test_folder, output_folder)


if __name__ == "__main__":
    main()


