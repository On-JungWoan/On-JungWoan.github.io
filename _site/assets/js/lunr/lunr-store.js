var store = [{
        "title": "DINO(2022) 논문 리뷰",
        "excerpt":"DINO: DETR with Improved DeNoising Anchor Boxes 3.1 Preliminaries __Conditional DETR과 DAB-DETR에서는 query를 positional part와 content part로 나누었으며, 본 논문에서는 각각을 positional query와 content query라 언급한다. 또한, DAB-DETR에서는 query를 $(x,y,w,h)$의 4D anchor box로 표현하였는데, 이는 decoder layer에서 anchor box를 refine하기 쉽게 하기 위함이다. 이 때 (x,y)는 box의 중심좌표를, (w,h)는 width와...","categories": ["DL_paper"],
        "tags": ["DL","computer_vision"],
        "url": "/dl_paper/dino/",
        "teaser": null
      },{
        "title": "Implementation of DINO(2022)",
        "excerpt":"1. Set Virtual Environment 가상환경 세팅은 conda를 사용하였으며, 개발 환경은 ubuntu / i9-13900K CPU / GTX 4080 / 64GB Mem이다. 1-1) Git Clone 우선, 아래의 DINO 공식 Github 링크에 들어가서 해당 repo를 local에 clone 해준다. link : https://github.com/IDEA-Research/DINO git clone https://github.com/IDEA-Research/DINO.git cd DINO 1-2) Setup Pytorch 그래픽 카드 버전에 맞는...","categories": ["DL_paper"],
        "tags": ["DL","computer_vision"],
        "url": "/dl_paper/dino_implements/",
        "teaser": null
      },{
        "title": "Covert onnx(NCHW) to tflite(NHWC)",
        "excerpt":"1. ONNX(NCHW)와 TFLite(NHWC)간의 Fomat문제 ONNX는 NCHW(채널, 높이, 너비) 형식의 이미지 데이터 포맷을 사용한다. 반면, TensorFlow Lite(TFLite)는 NHWC(높이, 너비, 채널) 형식의 이미지 데이터 포맷을 사용한다. 이러한 format 차이로 인해 onnx to tflite 변환 시 format issue가 발생한다. 이런 경우 직접 문제가 발생하는 layer를 찾아 shape을 수정해줘야만 한다. 본 포스팅에서는 onnx2tf를 사용하여...","categories": ["DL_optim"],
        "tags": ["DL","computer_vision"],
        "url": "/dl_optim/tflite/",
        "teaser": null
      },{
        "title": "Deformable DETR(2021) 논문 리뷰",
        "excerpt":"발표자료 : https://docs.google.com/presentation/d/1KFEG02jlgbZISuvFbilvwaP8PbdQCzAA/edit?usp=sharing&amp;ouid=116507288704586191771&amp;rtpof=true&amp;sd=true 발표영상 : Deformable DETR: Deformable Transformers for End-to-End Object Detection 리뷰 논문링크 : Deformable DETR: Deformable Transformers for End-to-End Object Detection Implementation : https://github.com/fundamentalvision/Deformable-DETR 1. Backgorund Deformable DETR에 대해 소개해드리기에 앞서, 먼저 선행 연구에 대해 소개하도록 하겠습니다. 1-1. Transformer Transformer는 Input contents와 Target contents간의 관계를 파악하여 attention...","categories": ["DL_paper"],
        "tags": ["DL","computer_vision"],
        "url": "/dl_paper/deform_detr/",
        "teaser": null
      },{
        "title": "Deformable DETR for edge device(초안)",
        "excerpt":"최근 발표된 SOTA 모델의 경우, 대부분 그 크기가 매우 크고 무겁습니다(특히 비전분야). 이러한 pre-trained 모델들을 개발 환경에서 사용할 때는 대부분 큰 문제가 되지 않습니다. 하지만 Android나 임베디드 보드와 같은 엣지 디바이스에서는 이러한 Full-size 모델을 사용하는 데 한계가 존재합니다. 따라서 torch 모델을 엣지 디바이스에서 사용하기 위해서는, 이를 최적화 된 포맷으로 변환해주는...","categories": ["DL_optim"],
        "tags": ["DL","computer_vision"],
        "url": "/dl_optim/ddetr_to_tflite/",
        "teaser": null
      },{
        "title": "VIBE(2020) 논문 리뷰 (작성중)",
        "excerpt":"발표자료 : 발표영상 : 논문링크 : VIBE: Video Inference for Human Body Pose and Shape Estimation Implementation : https://github.com/mkocabas/VIBE 1. Backgorund 1-1. SMPL 1-2. GAN 1-3. HMR 1-4. Temporal HMR 2. VIBE 2-0. Introduction 2-1. Architecture VIBE의 전체적인 모델 아키텍쳐는 HMR과 크게 다르지 않습니다. 우선, 길이 T의 input video V가...","categories": ["DL_paper"],
        "tags": ["DL","computer_vision"],
        "url": "/dl_paper/vibe/",
        "teaser": null
      }]
