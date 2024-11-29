import os
import cv2
import argparse
import yaml

import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm
from albumentations import Resize

from utils.utils_for_visualizer import encode_mask_to_rle
from segformer_to_esb import initialize_model


def parse_args():
    """
        터미널로 설정한 명령인자를 파싱
    """
    parser = argparse.ArgumentParser(description='Inference segmentation model')
    parser.add_argument('-c', '--config', type=str, default='smp_unetplusplus_efficientb0.yaml',
                        help='path to config file')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='threshold for binary prediction')
    args = parser.parse_args()

    return args


def load_config(config_name):
    """
        Config 파일 불러오기
    """
    config_path = os.path.join('configs', config_name)
    if not os.path.exists(config_path):
        print(f'Config 파일 없음: {config_path}')
        exit(1)

    with open(config_path, 'r') as f:
        try:
            config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            print(f'파일 불러오기 실패: {e}')
            exit(1)

    
    config.setdefault('image_root', '/data/ephemeral/home/data/train/DCM')
    config.setdefault('save_dir', '/data/ephemeral/home/repo/results')
    config.setdefault('output_name', 'ensemble_results.csv')
    config.setdefault('batch_size', 4)
    config.setdefault('num_workers', 2)
    config.setdefault('threshold', 0.5)
    
    return config



class EnsembleDataset(Dataset):
    def __init__(self, fnames, cfg, tf_dict):
        self.fnames = np.array(sorted(fnames))
        self.image_root = cfg['image_root']
        self.tf_dict = tf_dict
        self.ind2class = {i: v for i, v in enumerate(cfg['CLASSES'])}

    def __len__(self):
        return len(self.fnames)
    
    def __getitem__(self, item):
        image_name = self.fnames[item]
        image_path = os.path.join(self.image_root, image_name)
        image = cv2.imread(image_path)

        assert image is not None, f"{image_path} 해당 이미지를 찾지 못했습니다."
        
        image = image / 255.0
        return {"image": image, "image_name": image_name}

    def collate_fn(self, batch):
        images = [data['image'] for data in batch]
        image_names = [data['image_name'] for data in batch]
        inputs = {"images": images}

        image_dict = self._apply_transforms(inputs)
        image_dict = {k: torch.from_numpy(v.transpose(0, 3, 1, 2)).float()
                      for k, v in image_dict.items()}
        
        for image_size, image_batch in image_dict.items():
            assert len(image_batch.shape) == 4, \
                f"collate_fn 내부에서 image_batch의 차원은 반드시 4차원이어야 합니다.\n현재 shape : {image_batch.shape}"
            assert image_batch.shape == (len(batch), 3, image_size, image_size), \
                f"collate_fn 내부에서 image_batch의 shape은 ({len(batch)}, 3, {image_size}, {image_size})이어야 합니다.\n현재 shape : {image_batch.shape}"

        return image_dict, image_names
    
    def _apply_transforms(self, inputs):
        return {
            key: np.array(pipeline(**inputs)['images']) for key, pipeline in self.tf_dict.items()
        }


def load_models(cfg, device):
    model_dict = {}
    model_count = 0

    print("\n======== Model Load ========")
    for key, paths in cfg['model_paths'].items():
        if len(paths) == 0:
            continue
        model_dict[key] = []
        print(f"{key} image size 추론 모델 {len(paths)}개 불러오기 진행 시작")
        for path in paths:
            if os.path.basename(path) == 'segformer.pt':
                continue
            print(f"{os.path.basename(path)} 모델을 불러오는 중입니다..", end="\t")
            model = torch.load(path).to(device)
            model.eval()
            model_dict[key].append(model)
            model_count += 1
            print("불러오기 성공!")
        print()
    print(model_dict.keys())
    model = initialize_model( 
        model_name='segformer',  
        num_classes=29, 
        encoder_name='nvidia/mit-b5',
        pretrained=False, 
        encoder_weights=None,  # 
    ) 
    model_path = 'checkpoints/segformer.pt'
    model.load_state_dict(torch.load(model_path, map_location=device)) 
    # print(model)
    print(f"Model 'segformer' loaded from {model_path} and moved to {device}.") 
    model.eval()
    key = '/data/ephemeral/home/repo/checkpoints/segformer.pt'
    model_dict[key].append(model)
    model_count += 1

    print(f"모델 총 {model_count}개 불러오기 성공!\n")
    return model_dict, model_count


def save_results(cfg, filename_and_class, rles):
    classes, filename = zip(*[x.split("_") for x in filename_and_class])
    image_name = [os.path.basename(f) for f in filename]

    df = pd.DataFrame({
        "image_name": image_name,
        "class": classes,
        "rle": rles,
    })

    print("\n======== Save Output ========")
    print(f"{cfg['save_dir']} 폴더 내부에 {cfg['output_name']}을 생성합니다..", end="\t")
    os.makedirs(cfg['save_dir'], exist_ok=True)

    output_path = os.path.join(cfg['save_dir'], cfg['output_name'])
    try:
        df.to_csv(output_path, index=False)
    except Exception as e:
        print(f"{output_path}를 생성하는데 실패하였습니다.. : {e}")
        raise

    print(f"{os.path.join(cfg['save_dir'], cfg['output_name'])} 생성 완료")


def soft_voting(cfg):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    fnames = {
        os.path.relpath(os.path.join(root, fname), start=cfg['image_root'])
        for root, _, files in os.walk(cfg['image_root'])
        for fname in files
        if os.path.splitext(fname)[1].lower() == ".png"
    }

    tf_dict = {image_size: Resize(height=image_size, width=image_size) 
               for image_size, paths in cfg['model_paths'].items() 
               if len(paths) != 0}
    
    print("\n======== PipeLine 생성 ========")
    for k, v in tf_dict.items():
        print(f"{k} 사이즈는 {v} pipeline으로 처리됩니다.")

    dataset = EnsembleDataset(fnames, cfg, tf_dict)
    
    data_loader = DataLoader(dataset=dataset,
                             batch_size=cfg['batch_size'],
                             shuffle=False,
                             num_workers=cfg['num_workers'],
                             drop_last=False,
                             collate_fn=dataset.collate_fn)

    model_dict, model_count = load_models(cfg, device)
    print(model_dict)
    
    filename_and_class = []
    rles = []

    print("======== Soft Voting Start ========")
    with torch.no_grad():
        with tqdm(total=len(data_loader), desc="[Inference...]", disable=False) as pbar:
            for image_dict, image_names in data_loader:
                total_output = torch.zeros((cfg['batch_size'], len(cfg['CLASSES']), 2048, 2048)).to(device)
                for name, models in model_dict.items():
                    for model in models:
                        outputs = model(image_dict[name].to(device))
                        
                        # Handle dict output
                        if isinstance(outputs, dict):
                            outputs = outputs['out']  # Extract main output
                        
                        outputs = F.interpolate(outputs, size=(2048, 2048), mode="bilinear")
                        outputs = torch.sigmoid(outputs)
                        total_output += outputs

                total_output /= model_count
                total_output = (total_output > cfg['threshold']).detach().cpu().numpy()

                for output, image_name in zip(total_output, image_names):
                    for c, segm in enumerate(output):
                        rle = encode_mask_to_rle(segm)
                        rles.append(rle)
                        filename_and_class.append(f"{dataset.ind2class[c]}_{image_name}")
                
                pbar.update(1)


    save_results(cfg, filename_and_class, rles)


def main(args=None):
    if args is None:
        args = parse_args()

    cfg = load_config(args.config)
    soft_voting(cfg)


if __name__ == "__main__":
    main()

