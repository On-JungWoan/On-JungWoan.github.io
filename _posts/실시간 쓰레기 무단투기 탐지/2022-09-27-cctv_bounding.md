---
title:  "알고리즘 흐름 / CUDA Streams"
excerpt: "쓰레기 무단투기 탐지 프로젝트"

categories:
  - projects
tags:
  - "딥러닝"
  - "computer_vision"
  - "ICT인턴십"
  - "유클리드소프트"
  - "CUDA"

published: true

toc: true
toc_sticky: true
 
date: 2022-09-27
last_modified_at: 2022-09-28
til: 'true'
permalink: "/cctv/cctv_bounding/"
legacy_category: cctv
series: cctv
---

## 1. 객체/행위 탐지 알고리즘 흐름

![image](https://user-images.githubusercontent.com/84084372/192476175-0fb00fa9-3af7-4fe6-994c-c166ee80a39d.png)

jpg 이미지를 받으면 해당 이미지를 tensor로 변환하고, trt_model에 해당 tensor를 삽입한다. 다양한 사람이 이미지를 관찰하듯이 여러 case에서 해당 이미지의 객체를 인식하고, confidence가 가장 높은 바운딩 박스를 채택한다. 그 뒤, 해당 바운딩 박스보다 약간 더 크게 이미지를 crop하고(인식률 개선), key point를 추출한다. 마지막으로, 여러 조건식을 통해 투기 행위로 인식되면 box, skeleton, confidence를 이미지에 그린다. 이러한 알고리즘을 거쳐서 최종적으로 dumping_yn의 boolean값과 bounding box, skeleton, confidence가 그려진 이미지가 return된다.

<br>
<br>

## 2. CUDA Streams
### 2-1) 문제상황

코드 디버깅을 하던 중, 특정 코드를 지나면 bounding box의 정보를 담고있는 output값이 0이 되는 현상이 발생했다. 아래는 output에 대한 description인데, 0번 원소는 box의 개수이고, 그 뒤로 box의 정보가 6개씩 사이클 돌고 있는 (1, -1) shape의 array이다. 이러한 output을 1번 원소부터 (-1, 6)으로 reshape하여 열 개수를 6개로 고정하고, 0번 원소를 num으로 받아 num 개수만큼 행을 추출한다. 이게 bounding box의 정보가 되어 다음 로직이 순차적으로 실행이 되어야하는데 특정 코드를 만나면 output의 값이 0이나 이상한 매우 작은 값으로 변형되는 issue가 있었다.

```
output:     A numpy likes [num_boxes,cx,cy,w,h,conf,cls_id, cx,cy,w,h,conf,cls_id, ...]
```

<br>

### 2-2) 원인

해당 코드는 cpu(host)의 데이터를 gpu(device)에 올려 연산하는 코드였는데, gpu가 값 위에 계속 새로운 값을 연산하여 결국은 0이나 매우 작은 값이 나왔던 것이다. 결국 이미지가 적절하지 않은 이미지였고, 사람이 너무 커서 값이 이상하게 나온 것이라고 한다. 적절한 이미지는 차장님께서 다시 보내주신다고 하셨다.

<br>

### 2-3) 해당 CUDA 프로그래밍 패턴

- **1) 스트림 정의**

```python
cudart.cudaStreamCreate()
```

- **2) host, device input 값 할당**

```python
cudart.cudaMallocAsync()
```

- **3) Input Data를 Host에서 Device로 할당**

```python
cudart.cudaMemcpyAsync(cuda_input, host_input.ctypes, host_input.nbytes, cudaMemcpyHostToDevice, 정의한 stream)
```

- **4) 연산이 끝난 output을 다시 Host에 할당**

```python
cudart.cudaMemcpyAsync(host_output.ctypes, cuda_output, host_output.nbytes, cudaMemcpyDeviceToHost, 정의한 stream)
```                       

- **5) Synchronize the stream**

```python
cudart.cudaStreamSynchronize(stream)
```

<br>

## 3. Output Image

![image](https://user-images.githubusercontent.com/84084372/192480229-fc3afdea-0b86-4379-bfe7-adf734a27cf5.png)