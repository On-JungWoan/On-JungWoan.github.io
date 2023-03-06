---
title:  "Implementation of DINO(2022)"
excerpt: "DINO: DETR with Improved DeNoising Anchor Boxes for End-to-End Object Detection"

categories:
  - DL
tags:
  - [DL]

published: true

toc: true
toc_sticky: true
 
date: 2023-03-05
last_modified_at: 2023-03-05
use_math: true
---

## 1. Set Virtual Environment

가상환경 세팅은 conda를 사용하였으며, 개발 환경은 ubuntu / i9-13900K CPU / GTX 4080 / 64GB Mem이다.

![image](https://user-images.githubusercontent.com/84084372/222917401-9f057caf-c912-4db8-aab5-d40da1c64da0.png)

### 1-1. Git Clone

우선, 아래의 DINO 공식 Github 링크에 들어가서 해당 repo를 local에 clone 해준다.

> link : <https://github.com/IDEA-Research/DINO>

```
git clone https://github.com/IDEA-Research/DINO.git
cd DINO
```

<br>

### 1-2. Setup Pytorch

그래픽 카드 버전에 맞는 pytorch를 install 해준다. 4080은 어떤 버전을 사용해야 하는지 잘 몰라서 가장 최신 버전인 11.7 버전을 가상환경에 설치해주었다.

> link : <https://pytorch.org/get-started/locally>

![image](https://user-images.githubusercontent.com/84084372/222917489-cf2632cf-72ee-4a3d-88d4-0adce5df7773.png)

torch.cuda.is_available()의 return값이 True이면 버전에 맞게 잘 설치된 것이다.

```
>>> import torch
>>> torch.cuda.is_available()
True 
```

- **주의사항**

  만약, pytorch build가 cpu로 설치됐다면, 버전이 맞지 않는 것이므로 다른 버전을 찾아 설치해주면 된다. 버전이 맞지 않으면 cuda를 사용할 수 없으니 버전을 잘 맞추도록 하자

  ![image](https://user-images.githubusercontent.com/84084372/222917546-d59620db-de5e-431e-80d8-e81673fddb2c.png)


<br>

### 1-3. requirements 설치

#### 1-3-1. install requirements.txt

```
pip install -r requirements.txt
```

#### 1-3-2. Compiling CUDA operators
```
cd DINO/models/dino/ops
python setup.py build install
# unit test (should see all checking is True)
python test.py
cd ../../..
```

<br>
<br>

## 2. Prepare Dataset

Dataset은 다음과 같은 구조로 설치하면 된다.

```
COCODIR/
  ├── train2017/
  ├── val2017/
  └── annotations/
  	├── instances_train2017.json
  	└── instances_val2017.json
```

터미널에 아래의 명령어를 차례차례 입력하면 된다. 해외 서버에서 wget으로 받아오다보니 시간이 오래 걸린다. 조금 더 빨리 받아올 수 있는 방법이 있는걸로 알고있는데 정확히 기억나지 않아서 그냥 기다렸다.

```
cd DINO
mkdir COCODIR
cd COCODIR

wget http://images.cocodataset.org/zips/train2017.zip
wget http://images.cocodataset.org/zips/val2017.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip

unzip train2017.zip
unzip val2017.zip
unzip annotations_trainval2017.zip

rm train2017.zip
rm val2017.zip
rm annotations_trainval2017.zip
```

<br>
<br>

## 3. Pre-trained Model Inference

공식 github에서 Model Zoo를 제공하고 있어, 다양한 세팅에 대해 Inference 및 evaluation을 해볼 수 있었다. 현재는 간단한 inference만 해보면 되기 때문에, 4 scale feature로 12 epoch 학습(Resnet50 백본)한 모델의 체크 포인트를 사용하였다.

> ckpts link : <https://drive.google.com/file/d/1eeAHgu-fzp28PGdIjeLe-pzGPMG2r2G_/view?usp=sharing>

> code link : <https://github.com/On-JungWoan/DINO-2022-implement/blob/main/inference_and_visualization.ipynb>

Inference 코드는 다음과 같다. DINO 폴더 최상위에 작성하면 된다.

```python
import torch
import json
import datasets.transforms as T

from main import build_model_main
from datasets import build_dataset
from util.visualizer import COCOVisualizer
from util.slconfig import SLConfig
from PIL import Image
from util import box_ops

# PATH 지정
model_config_path = "config/DINO/DINO_4scale.py"
model_checkpoint_path = "ckpts/checkpoint0011_4scale.pth" # 방금 다운받은 체크포인트의 경로를 입력


# Model Build
# model을 build하고 가중치를 불러오는 과정
args = SLConfig.fromfile(model_config_path)
args.device = 'cuda'
model, _, postprocessors = build_model_main(args)
checkpoint = torch.load(model_checkpoint_path, map_location='cpu')
model.load_state_dict(checkpoint['model'])


# Load coco names
# id와 name을 매칭시키기 위해 json 파일을 불러오는 과정
with open('util/coco_id2name.json') as f:
    id2name = json.load(f)
    id2name = {int(k):v for k,v in id2name.items()}


# Load Coco Dataset
args.dataset_file = 'coco'
args.coco_path = "COCODIR/" # 본인의 COCODIR 입력
args.fix_size = False


# Load Sample Images
img_path = "./figs/test.jpg" # inference 하길 원하는 이미지의 경로 입력
image = Image.open(img_path).convert("RGB")
transform = T.Compose([
    T.RandomResize([800], max_size=1333),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
image, _ = transform(image, None)


# Inference
output = model.cuda()(image[None].cuda())
output = postprocessors['bbox'](output, torch.Tensor([[1.0, 1.0]]).cuda())[0]


# visualize outputs
# box 좌표는 normalize 되어 있기 때문에 후처리 과정이 필요함
thershold = 0.3 # set a thershold

# thereshold 미만인 bbox는 걸러냄
scores = output['scores']
labels = output['labels']
boxes = box_ops.box_xyxy_to_cxcywh(output['boxes'])
select_mask = scores > thershold

# id to name 변환 및 bbox de-normalize
box_label = [id2name[int(item)] for item in labels[select_mask]]
pred_dict = {
    'boxes': boxes[select_mask],
    'size': torch.Tensor([image.shape[1], image.shape[2]]),
    'box_label': box_label
}

# Visualization
vslzr = COCOVisualizer()
vslzr.visualize(image, pred_dict, savedir=None, dpi=100)
```

<br>

### 3.1 결과

![output](https://user-images.githubusercontent.com/84084372/222920350-44c15ede-2d0f-4f17-bba7-0d444eca8134.png)

<br>
<br>

## 4. Train Model

DINO 모델 Train 관련 내용 적기
argument 관련 디버깅
시각화 내용들