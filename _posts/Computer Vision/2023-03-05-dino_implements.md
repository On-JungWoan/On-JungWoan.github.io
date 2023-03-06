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
---

$test$

<details>

<summary>test</summary>

```python
print(test)
```

</details>

## 1. Set Virtual Environment

가상환경 세팅은 conda를 사용하였으며, 개발 환경은 ubuntu / i9-13900K CPU / GTX 4080 / 64GB Mem이다.

![image](https://user-images.githubusercontent.com/84084372/222917401-9f057caf-c912-4db8-aab5-d40da1c64da0.png)

### 1-1) Git Clone

우선, 아래의 DINO 공식 Github 링크에 들어가서 해당 repo를 local에 clone 해준다.

> link : <https://github.com/IDEA-Research/DINO>

```
git clone https://github.com/IDEA-Research/DINO.git
cd DINO
```

<br>

### 1-2) Setup Pytorch

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

### 1-3) requirements 설치

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

공식 github에서 Model Zoo를 제공하고 있어, 다양한 세팅에 대해 Inference 및 evaluation을 해볼 수 있었다. 현재는 간단한 inference만 해보면 되기 때문에, 4 scale feature로 12 epoch 학습(Resnet50 백본)한 모델의 체크 포인트를 사용하였다. ipynb 코드는 아래 링크를 참조하면 된다.

> ckpts link : <https://drive.google.com/file/d/1eeAHgu-fzp28PGdIjeLe-pzGPMG2r2G_/view?usp=sharing>

> code link : <https://github.com/On-JungWoan/DINO-2022-implement/blob/main/inference_and_visualization.ipynb>

Inference 코드는 다음과 같다. DINO 폴더 최상위에 작성하면 된다.

<details>

<summary>코드 접기/펼치기</summary>

{% raw %}

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
import numpy as np

#
model_config_path = "config/DINO/DINO_4scale.py"
model_checkpoint_path = "logs/DINO/train_test/checkpoint_best_regular.pth" # your ckp path
img_dir = "figs/idea.jpg" # your image path


#
args = SLConfig.fromfile(model_config_path)
args.device = 'cuda'
model, criterion, postprocessors = build_model_main(args)
checkpoint = torch.load(model_checkpoint_path, map_location='cpu')
model.load_state_dict(checkpoint['model'])
_ = model.eval()



# load coco names
with open('util/coco_id2name.json') as f:
    id2name = json.load(f)
    id2name = {int(k):v for k,v in id2name.items()}



# 
image = Image.open(img_dir).convert("RGB")
transform = T.Compose([
    T.RandomResize([800], max_size=1333),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
image, _ = transform(image, None)


#
output = model.cuda()(image[None].cuda())
output = postprocessors['bbox'](output, torch.Tensor([[1.0, 1.0]]).cuda())[0]


# visualize outputs
if output['scores'].max() > 0.3:
    thershold = 0.3 # set a thershold
else:
    thershold = float(sorted(output['scores'].cpu())[-6])

np.array(output['scores'].cpu())

vslzr = COCOVisualizer()

scores = output['scores']
labels = output['labels']
boxes = box_ops.box_xyxy_to_cxcywh(output['boxes'])
select_mask = scores > thershold

box_label = [id2name[int(item)] for item in labels[select_mask]]
pred_dict = {
    'boxes': boxes[select_mask],
    'size': torch.Tensor([image.shape[1], image.shape[2]]),
    'box_label': box_label
}
vslzr.visualize(image, pred_dict, savedir=None, dpi=100)
```

{% endraw %}

</details>

<br>

### 3.1) 결과

기존 DETR 계열의 문제점이었던 작은 obj도 잘 감지하는 모습을 확인할 수 있다. 또한, obj가 겹쳐있는 경우도 문제없이 잘 추론하고 있다.

![output](https://user-images.githubusercontent.com/84084372/222920350-44c15ede-2d0f-4f17-bba7-0d444eca8134.png)

<br>
<br>

## 4. Train Model

COCO 데이터셋을 전부 사용하여 학습하기에는 시간이 다소 오래 걸릴 것 같아서 일부만 사용하였다. 다음은 예상 학습 시간을 계산한 테이블이다(rs50 backbone, 4scale 기준). 본인의 여건에 맞춰서 선택하면 된다. 해당 시간은 train 시간만 고려하였으므로 실제 train 시간은 아래 시간보다 더 오래 소요되며, 개발 환경에 따라 달라질 수 있다. 본 포스팅에서는 1만개 train dataset을 사용하여 12 epoch 학습하였다. 또한, 한 epoch 내에서의 loss 변화를 보기 위해 이미지 30장마다 loss를 기록해주었다. 

. | full dataset | 25,000 | 10,000 | 5000
:--: | :--: | :--: | :--: | :--: |
1epoch | 3h | 1.5h | 36m | 18m
12epoch | 36h | 18h | 7.2h | 3.6h

이를 위해 간단한 코드 custom을 해주었다.

<details>
<summary>코드 접기/펼치기</summary>

<div align="center"><strong>[Terminal]</strong></div>

```
# $1 : COCO Dir.
# $2 : Num of train_dataset
# $3 : Num of val_dataset
# $4 : use custom logger

bash scripts/DINO_train_custom.sh COCODIR/ 10000 5000 --custom_logger
```

<div align="center"><strong>[DINO_train_custom.sh]</strong></div>

```
coco_path=$1
python main.py \
    --num_train $2 --num_val $3 $4\
	--output_dir logs/DINO/train_$2_$3_4scale_rs50_12epc \
    -c config/DINO/DINO_4scale.py --coco_path $coco_path \
	--options dn_scalar=100 embed_init_tgt=TRUE \
	dn_label_coef=1.0 dn_bbox_coef=1.0 use_ema=False \
	dn_box_noise_scale=1.0
```

<div align="center"><strong>[main.py]</strong></div>

```python
...

parser.add_argument("--num_train", type=int)
parser.add_argument("--num_val", type=int)
parser.add_argument("--custom_logger", action='store_true')

...
```

<div align="center"><strong>[engine.py]</strong></div>

```python
...

if args.custom_logger:
  if _cnt%30 == 0:
      with open(args.output_dir + '/loss_only.txt', 'a') as f:
          f.write(f'{_cnt} : {loss_value}\n')

...          
```
</details>

<br>자세한 코드는 아래를 참고.

> link : <https://github.com/On-JungWoan/DINO-2022-implement>

<br>

### 4-1) Effectiveness

#### 4-1-1. Loss in Entire Epoch

- **For all epoch**

  ![image](https://user-images.githubusercontent.com/84084372/223036898-09877e9f-78b9-4bcb-bc20-0a46ce62717d.png)

- **Only 1, 6, 12 epoch**

  ![image](https://user-images.githubusercontent.com/84084372/223036917-b2b22657-82f2-4a8f-8530-8910751523eb.png)

#### 4-1-2. Loss (Mean of Epoch)

![image](https://user-images.githubusercontent.com/84084372/223036930-4e07697b-7e75-43eb-b3a4-6b5941dcc6ad.png)

#### 4-1-3. AP

![image](https://user-images.githubusercontent.com/84084372/223036949-d4085734-2df7-40fe-8768-222db79f5ba6.png)

#### 4-1-4. Epoch Time

![image](https://user-images.githubusercontent.com/84084372/223036966-320ac79a-3660-41dc-8213-4d2e5a73b61d.png)

<br>

### 4-2) Performance

#### 4-2-1. 1 Epoch model

![image](https://user-images.githubusercontent.com/84084372/223037065-2795ad67-70cb-4b50-8c48-4f10ffc7880b.png)

#### 4-2-2. 6 Epoch model

![image](https://user-images.githubusercontent.com/84084372/223037091-9cd730a3-6af9-43bb-903d-cd856751b7dc.png)

#### 4-2-3. 12 Epoch model

![image](https://user-images.githubusercontent.com/84084372/223037102-244e7f7d-33d3-4368-b847-80762676f9b1.png)

#### 4-2-4. Full Dataset model

![image](https://user-images.githubusercontent.com/84084372/223037201-8ab7eaaa-06be-447d-ab28-d74d6dbf28b8.png)
