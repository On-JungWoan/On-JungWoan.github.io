var store = [{
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
        "title": "Deformable DETR(2021) 논문 정리 (진행중)",
        "excerpt":"Abstract DETR은 obj detection에서 좋은 performance를 보여줌과 동시에 많은 hand-desinged componets를 제거함으로써 완전한 end-to-end의 학습을 할 수 있게 되었습니다. 그러나, Trnasformer attention module의 한계로 인해 DETR에는 다음과 같은 2가지 문제가 존재합니다. Slow convergence Limited feature spatial resolution 본 저자는 이러한 문제를 해결하기 위해 Deformable DETR을 제안합니다. Deformable DETR의 attention module은...","categories": ["DL_paper"],
        "tags": ["DL","computer_vision"],
        "url": "/dl_paper/deform_detr/",
        "teaser": null
      }]
