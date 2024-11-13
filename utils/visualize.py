import os
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import random

os.chdir(os.path.dirname(os.path.abspath(__file__)))

def visualize_and_save(dcm_folder, json_folder, output_base_folder):
    for id_folder in os.listdir(dcm_folder):
        image_dir = os.path.join(dcm_folder, id_folder)
        json_dir = os.path.join(json_folder, id_folder)
        output_dir = os.path.join(output_base_folder, id_folder)

        if os.path.isdir(image_dir) and os.path.isdir(json_dir):
            os.makedirs(output_dir, exist_ok=True)  

            for image_file in os.listdir(image_dir):
                if image_file.endswith('.png'):
                    image_path = os.path.join(image_dir, image_file)
                    json_file = image_file.replace('.png', '.json')
                    json_path = os.path.join(json_dir, json_file)

                    if os.path.exists(json_path):
                        output_path = os.path.join(output_dir, image_file)
                        image = Image.open(image_path)
                        fig, ax = plt.subplots(1)
                        ax.imshow(image)

                        with open(json_path, 'r') as f:
                            data = json.load(f)
                            
                        for item in data['annotations']:
                            points = item['points']
                            color = [random.random() for _ in range(3)]
                            polygon = patches.Polygon(points, closed=True, fill=True, edgecolor=None, facecolor=color, alpha=0.3)
                            ax.add_patch(polygon)

                        plt.axis('off')
                        plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
                        plt.close()

def main():
    dcm_folder = "../../data/train/DCM"
    json_folder = "../../data/train/outputs_json"
    
    #---------------------------------- train 데이터 시각화 ----------------------------------# 
    visualize_and_save_output_folder = "../../img/train_visualized"
    visualize_and_save(dcm_folder, json_folder, visualize_and_save_output_folder)
    #----------------------------------------------------------------------------------------#

if __name__ == "__main__":
    main()