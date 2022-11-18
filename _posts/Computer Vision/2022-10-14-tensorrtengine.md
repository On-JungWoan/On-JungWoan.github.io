---
title:  "TensorRT 모델에서 engine 모듈 추출하기"
excerpt: "pth 확장자 TensorRT 모델에서 engine부 추출"

categories:
  - DL
tags:
  - [DL, TensorRT, pytorch, ICT인턴십]

published: true

toc: true
toc_sticky: true
 
date: 2022-10-14
last_modified_at: 2022-10-14
til: 'true'
---


## 1. Make Engine File

pytorch 모델이 onnx를 거치지 않고, torch2trt를 사용해 바로 TensorRT로 변환되면 확장자가 pth가 된다. `~_trt.pth`의 TensorRT 모델에서 engine파일을 추출하는 과정이 잘 안되어서 삽질하다가 얻은 결과이다. 방법은 총 2가지가 있다.

### 방법1 : state_dict에서 추출

TorchRT 모델의 state_dict에서 engine을 추출하는 방법이다. 추출된 engine 파일을 serialize해서 이진파일로 저장하고, 다시 불러올 때는 deserialize해주면 된다.

```python
TRT_LOGGER = trt.Logger(trt.Logger.INFO)
runtime = trt.Runtime(TRT_LOGGER)

def _create_engine(model_path: str): 
        state_dict = torch.load(model_path, map_location='cpu')
        engine_bytes = state_dict['engine']
        with trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(engine_bytes)

with open(save_path, "wb") as f:
    f.write(_create_engine(model_path).serialize())
```

<br>

### 방법2 : TensorRT 모델에서 추출

TensorRT 모델에서 직접 engine file을 추출하는 방법이다. 훨씬 간단하다. 마찬가지로 불러올때는 deserialize 해주면 된다.

```python
model_trt = TRTModule()
model_trt.load_state_dict(torch.load(OPTIMIZED_MODEL))
TensorRT_model = model_trt.eval()

with open(ENGINE_PATH, "wb") as f:
    f.write(TensorRT_model.engine.serialize())
```

<br>

### 방법1 vs 방법2

둘 중 어떤 방식이 더 좋을지 차장님께 여쭤보았다. 우선 sate_dict에서 engine을 뽑아내는 방식은 runtime 시간이 있어서, model에서 직접 뽑는 것보다 조금 더 시간이 오래걸린다고 하셨다. 따라서 TensorRT 모델에서 직접 engine file을 뽑는 것이 더 유리할 것 같다.

<br>
<br>

## 2. Inference

먼저, engine 파일을 불러와서 deserialize 해준다. engine을 실행시키기 위해 engine.create_execution_context()까지 생성해주면 engine을 불러오는 과정은 끝이 난다. 그 이후 engine 모듈을 Stream으로 묶고, 각 텐서들을 GPU에 할당하면 된다. CUDA Stream에 대한 과정은 내 기술 블로그에 정리되어있다.

```python
with open(engine_file_path, "rb") as f:
    engine_data = f.read()

engine = runtime.deserialize_cuda_engine(engine_data)

context = engine.create_execution_context()

(...)
```

- **Refer.**

  [알고리즘 흐름 / CUDA Streams](https://on-jungwoan.github.io/cctv/cctv_bounding/#2-cuda-streams)

<br>
<br>


## 3. TensorRT vs TensorRT Engine?

내가 찾아본 바로는 TensorRT는 Optimizer부와 Runtime Engine부로 구성된다고 한다. Optimizer부는 모델을 최적화 하는 역할을 하고 Runtime Engine부에서는 다양한 GPU에서 연산을 한다. 여기서 Runtime Engine만 떼어낸 것이 TensorRT Engine 파일인데, 그래서 이 엔진파일을 사용하면 무엇이 좋을까? 우선, 일반 Pytorch 모델과 TensorRT 모델과는 확실한 속도차이가 존재했지만, TensorRT와 TensorRT Engine간에는 큰 속도 차이가 존재하지 않았다. 따라서 해당 부분은 차장님께 여쭤보았다.

<br>

우선, 일반 TensorRT 같은 경우는, 모델 프로젝트 파일과 함께 돌돌 말아놓은 형태여서, 모델 파일에 의존한다고 하셨다. 이 말인 즉슨, 모델 파일이 조금이라도 손상되면 TensorRT 파일도 사용할 수 없다는 것이다. 반면 Engine 파일은 그러한 종속성이 없어진 상태여서 파인튜닝할 때 주로 사용하신다고 하였다. 사실 TensorRT에 대한 자료는 너무 적어서 한국어로 된 자료가 거의 없다. 그래서 차장님도 확실하지는 않다고 하셨다. 개인 프로젝트를 할 때는 그냥 TensorRT만 사용하면 될 것 같다.

<br>
<br>

- **Reference**

  [https://github.com/NVIDIA-AI-IOT/torch2trt/issues/254](https://github.com/NVIDIA-AI-IOT/torch2trt/issues/254)

  [Developer Guide :: NVIDIA Deep Learning TensorRT Documentation](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html)
