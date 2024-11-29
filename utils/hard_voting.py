import numpy as np
import pandas as pd
import os
from tqdm import tqdm

def csv_ensemble(csv_paths, save_dir, threshold):
    def decode_rle_to_mask(rle, height, width):
        s = rle.split()
        starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
        starts -= 1
        ends = starts + lengths
        img = np.zeros(height * width, dtype=np.uint8)
        
        for lo, hi in zip(starts, ends):
            img[lo:hi] = 1
        
        return img.reshape(height, width)

    def encode_mask_to_rle(mask):
        pixels = mask.flatten()
        pixels = np.concatenate([[0], pixels, [0]])
        runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
        runs[1::2] -= runs[::2]
        return ' '.join(str(x) for x in runs)

    csv_column = 8352

    csv_data = []
    for path in csv_paths:
        data = pd.read_csv(path)
        csv_data.append(data)

    file_num = len(csv_data)
    filename_and_class = []
    rles = []

    print(f"앙상블할 모델 수: {file_num}, threshold: {threshold}")

    for index in tqdm(range(csv_column)):    
        model_rles = []
        for data in csv_data:
            if(type(data.iloc[index]['rle']) == float):
                model_rles.append(np.zeros((2048,2048)))
                continue
            model_rles.append(decode_rle_to_mask(data.iloc[index]['rle'],2048,2048))
        
        image = np.zeros((2048,2048))

        for model in model_rles:
            image += model
        
        image[image <= threshold] = 0
        image[image > threshold] = 1

        result_image = image

        rles.append(encode_mask_to_rle(result_image))
        filename_and_class.append(f"{csv_data[0].iloc[index]['class']}_{csv_data[0].iloc[index]['image_name']}")

    classes, filename = zip(*[x.split("_") for x in filename_and_class])
    image_name = [os.path.basename(f) for f in filename]

    df = pd.DataFrame({
        "image_name": image_name,
        "class": classes,
        "rle": rles,
    })

    df.to_csv(save_dir, index=False)

if __name__ == "__main__":
    csv_paths = [
        'csv 경로1', 'csv 경로2', 'csv 경로3', 'csv 경로4', 'csv 경로5'
    ]
    for threshold in [2]: # threshold 값 변경
        save_path = f"ensemble_threshold_{threshold}.csv"
        csv_ensemble(csv_paths, save_path, threshold)