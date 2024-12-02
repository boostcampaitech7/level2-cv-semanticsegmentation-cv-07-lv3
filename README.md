# **Hand Bone Image Segmentation**

<p align="center">
<img width="1235" alt="image" src="https://github.com/user-attachments/assets/9a50d862-72fd-4e65-9b9b-4271a7a016d8" width="90%" height="90%"/>
</p>

## 1. Competiton Info

  본 대회는 X-ray 이미지에서 사람의 손 뼈를 Segmentation 하는 대회이다. 데이터셋은 Ground Truth로 29개 클래스에 해당하는 Mask가 포함된 X-ray 형태의 PNG 이미지가 제공되었다. 평가지표로는 Dice Coefficient가 사용됐다.

- **Input :** X-ray 형태의 사람 손 뼈에 해당하는 PNG 이미지
- **Output :** 모델이 예측한 각 픽셀 좌표의 Class와 Points를 RLE로 변환한 값
- **손가락, 손등, 팔 뼈 총 29개 클래스**
    - finger-1~19, Trapezium, Trapezoid, Capitate, Hamate, Scaphoid, Lunate, Triquetrum, Pisiform, Radius, Ulna

### Timeline

- 2024.11.11 ~ 2024.11.28

### Evaluation

- 평가지표: Dice Coefficient

## 2. Team Info

### MEMBERS

| <img src="https://github.com/user-attachments/assets/51235bb8-bff2-4026-a2af-12e4eb78fe66" width="200" height="200"/> | <img src="https://github.com/user-attachments/assets/ba0a59b9-88db-43a1-995c-977a85c3128f" width="200" height="200"/> | <img src="https://github.com/user-attachments/assets/b9eee66f-51db-4dbb-b61f-f2ea12956da0" width="200" height="200"/> | <img src="https://github.com/user-attachments/assets/72df8b46-718a-4471-b176-ba17e3887a77" width="200" height="200"/> | <img src="https://github.com/user-attachments/assets/7afc3745-190d-426c-8f69-5b47fce5c0e1" width="200" height="200"/> | <img src="https://github.com/user-attachments/assets/949fd85a-0a55-435b-8a36-20d98eac7ce4" width="200" height="200"/> |
| --- | --- | --- | --- | --- | --- |
| [김민솔](https://github.com/kim-minsol) | [김예진](https://github.com/yeyechu) | [배형준](https://github.com/BaeHyungJoon) | [송재현](https://github.com/mongsam2) | [이재효](https://github.com/jxxhyo) | [차성연](https://github.com/MICHAA4) |

### Project Objective and Direction

- Git을 이용하여 공통된 템플릿 코드를 사용하고, 체계적인 프로젝트 버전을 관리하였다.
- 서버 사용률을 높일 수 있도록 실험 일정을 Notion으로 공유하여 관리하였다.
- 실험의 목적을 달성하기 위해 독립 변인을 하나씩만 설정하여 각 요인이 모델 성능에 미치는 영향을 명확히 파악하고 분석할 수 있도록 하였다.

### Team Component

- **코드 작성 :** (SMP Baseline Template) 배형준, (AutoSam) 김민솔, (시각화) 김예진
- **EDA와 데이터 전처리** : (EDA) 김예진, 송재현, 이재효, 차성연, (전처리) 김예진, 차성연, (K-Fold) 이재효
- **하이퍼파라미터 실험 :** (증강) 김예진, (손실 함수) 김예진, 차성연, (최적화, 옵티마이저, 스케줄러) 이재효
- **모델 실험 :** (모델) 김민솔, 배형준, 차성연, 이재효

## 3. Data EDA

<img src="https://github.com/user-attachments/assets/764e6d89-a414-435e-a8a6-5127a1c3e401" width="600" height="400">

- 전체 이미지에 29개 클래스가 균일하게 포함되었다
- 전체 이미지에서 손 끝 마디뼈인 finger-1, 4, 8, 12, 16과 손등 뼈 중에는 Trapezoid와 Pisiform의 픽셀 비율이 가장 적었으며, 팔 뼈인 Radius와 Ulna가 가장 큰 면적을 차지했다.
- 꺾인 손목 데이터 비율은 학습(11.5%)보다 테스트(57.6%)에서 더 높고, 메타데이터에 값을추가하여 관리하였다
- Multi-label 데이터셋에서 Pisiform과 Triquetrum, 그리고 Trapezium과 Trapezoid의 손등 뼈가 일부 겹치는 영역이 가장 크게 나타났다

## 3. Data Augmentation

- Resize + CropNonEmptyMaskIfExists
- CLAHE
- RandomRotation

## 4. Model

- UNet++_EfficientNet b0
- UNet++_EfficientNet b7
- UNet++_MaxViT
- UNet++_HRNet
- AutoSAM

## 5. Result

- Soft Voting Ensemble (0.9740)

| EfficientNet b7 Fold 0 | HRNet Fold 0 | HRNet Fold 1 | HRNet Fold 2 | HRNet Fold 3 | HRNet Fold 4 |
| --- | --- | --- | --- | --- | --- |

### Feedback

- 프로젝트 목표에 맞게 주어진 Hand-bone X-Ray 데이터에서 큰 사이즈의 뼈부터 작은 사이즈의 뼈까지 다양한 크기의 물체를 검출하는 모델 파이프라인을 구성하였다.
- Git / GitHub / Notion / Slack 협업 툴을 잘 활용하여 원할하게 실험 공유를 가능하게 되어서 실험에 대한 피드백을 확실하게 주고 받을 수 있었다. 또한 이어지는 실험에 대한 방향성도 더 쉽게 잡을 수 있었다.

## 5. Report

- [Wrap-up Report](https://www.notion.so/Hand-Bone-Image-Segmentation-87b631c736ba4e1d82c3f636bddbfbd8?pvs=21)

## 6. Project File Structure

```
repo/
├── AutoSAM/ 
├── configs/
│   └── torchvision_fcn_resnet50.yaml          
│   └── smp_unetplusplus_efficientb0.yaml
├── eda/
│   └── eda.ipynb
│   └── calculate.py  
├── src/
│   ├── dataset/
│   │   └── dataset.py       
│   ├── models/
│   │   └── torchvision_models.py         
│   │   └── smp_models.py                 
│   ├── loss.py          
│   ├── optimizer.py  
│   ├── scheduler.py    
│   └── trainer.py          
├── utils/
│   └── collect_and_save.py
│   └── convert_excel_to_csv.py
│   └── format.py
│   └── gt_visualize.py
│   └── labelme.py
│   └── resources.py
│   └── test_visualize.py
│   └── train_visualize.py
│   └── utils_for_visualizer.py
│   └── val_visualize.py
│   └── val_each_class.py 
├── README.md
├── hard_ensemble.py            
├── inference.py    
├── main.py
├── soft_ensemble.py
└── train.py      
```

## 7. How to run

### Train & Save Checkpoint

`python train.py -c {생성한 config 이름}.yaml`

### Inference

`python inference.py -c {생성한 config 이름}.yaml -m {추론할 model checkpoint}.pt`

### Main process (Train + Inference)

`python main.py -c {생성한 config 이름}.yaml`
