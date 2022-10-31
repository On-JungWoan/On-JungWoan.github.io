---
title:  "[실시간 영상처리 최적화] CUDA Stream"
excerpt: "Null Stream / Non-Null Stream"

categories:
  - DL
tags:
  - [DL, ICT인턴십]

published: true

toc: true
toc_sticky: true
 
date: 2022-10-31
last_modified_at: 2022-10-31
---

## 1. Stream
### 1-0) Stream이란?
host에서 device로 명령을 보내는 통로이다. host에서 stream으로 명령을 넣어주면 device는 그걸 꺼내서 사용한다.

<br>

### 1-1) NULL Stream
사용할 stream을 명시하지 않으면 NULL stream이 사용된다. NULL Stream 또는, 하나의 NON-NULL stream에 들어온 명령은 순서대로(동기적) device에 의해 처리된다.

<p align="center"><img src="https://user-images.githubusercontent.com/84084372/198909088-1cc811ae-0540-4371-918f-33fca218181c.png"></p>

<br>

### 1-2) NON-NULL Stream
명시적으로 생성 및 사용한 stream을 말한다. 서로 다른 stream 안에 있는 명령들 사이의 실행 순서는 비동기적(정해져있지 않음)이며, 실행 순서는 GPU의 스케쥴링 기법에 따라서 달라진다.

<p align="center"><img src="https://user-images.githubusercontent.com/84084372/198909110-8309602b-151c-417b-83b9-e75e19516315.png"></p>

비동기적 연산을 위해서는, 명시적으로 사용할 만큼의 non-null stream을 만들어서 사용하며, NON-NULL Stream의 작업 중 일부는 동시에 수행이 가능하다. 동시에 수행 가능한 작업은 다음과 같다.

- **Host Computation and Device Computation**
- **Host Computation and Host-Device Data Transfer**
- **Device Computation and Host-Device Data Transfer**
- **Concurrent Device Computations**

따라서, NON-NULL Stream을 이용함으로써 Host 연산 중 memory copy 및 kernel 연산이 동시에 이루어질 수 있으며, 서로 다른 커널들의 연산도 동시에 시행될 수 있다.

<br>
<br>

## 2. Multi-core Programming by NON-NULL Stream

### 2-1) NULL Stream
NULL Stream을 사용한 연산은 동기적이다. 따라서 Host에서 Device로 Memory Copy가 진행되는 동안, Kernal과 D2H는 아무런 동작을 하지 않고 놀게된다. H2D 메모리 카피가 끝나고 커널 연산을 하는 동안 또 다시 노는 부분이 생기게 된다. 이러한 비효율적인 부분을 해결하기 위해 NON-NULL Stream을 사용할 수 있다.

<br>

### 2-2) NON-NULL Stream
NON-NULL Stream은 연산 결과를 한번에 보내는게 아니라, 각각의 task를 나누어서 연산이 되는 것부터 먼저 보낸다. 이렇게 잘라진 것들을 여러개의 NON-NULL Stream으로 나누어 보내면, 노는 부분이 없이 동시에 병렬 연산이 가능하다.

<br>

### 2-3) Example

1개의 NULL Stream 및 3개의 NON-NULL Stream 사용 예제이다.

<p align="center"><img src="https://user-images.githubusercontent.com/84084372/198910036-9cc506a2-4a30-4331-9a99-61a7b31ca588.png"></p>

1. **H2D_1(H2D의 첫 번째 조각) 명령을 먼저 수행하여 device로 메모리 전송**
2. **K_1(첫 번째 조각에 대한 커널 연산)을 수행한다. 또한, 동시에 H2D 통로가 쉬고 있으므로 H2D_2(2번째 H2D 작업 영역) 명령을 수행하여 device로 메모리 전송**
3. **K_1 연산 결과를 D2H_1을 통해 host로 보내주면, 다시 커널이 사용 가능해지므로 K_2 연산 수행. 또한 동시에 H2D_3를 통해 device로 메모리 전송**
4. **K_2 연산결과를 D2H_2 명령으로 다시 host로 넘겨줌. 동시에 k_3 연산 수행**
5. **마지막 K_3 연산 결과를 D2H_3 명령으로 다시 host에 넘겨줌**

<br>

### 2-4) 결론

이렇게 작업을 분할하여 여러 스트림에 분할함으로써, 연산 시간을 크게 줄일 수 있다. GPU 사용시, 데이터 보내는 작업이 상당히 큰 오버해드인데, NON-NULL Stream을 사용하면 이를 숨기면서 GPU가 계속 돌 수 있게 할 수 있다.

<br>
<br>

## 3. 비동기적 데이터 전송방법
### 3-1) Stream 생성

사용할 stream을 생성해준다.

```python
_, stream = cudart.cudaStreamCreate() # 제거 : cudaStreamDestroy()
```

<br>

### 3-2) Pinned Memory 생성

```
cudaMallocHost -> host에 메모리를 잡을 것인데, 이를 cuda가 잡겠다는 뜻
cudaFreeHost -> 풀어주는 것
```

host와 device가 일하는 동안에 데이터를 보내고 싶다면, 반드시 그 데이터를 보내거나 받는 영역을 pinned memory로 잡아야 한다.

- **pinned memory?**

    가상 메모리(swap device) 사용 시, swap device에 프로세스들이 페이지 단위로 구분되어 있으며, 이 중 필요한 page만 메모리에 올라와서 사용된다. 필요 없으면 다시 swap device로 내려가는데, 이를 swap out이라고 한다. 비동기적 데이터 전송을 위해서는 해당 메모리가 swap out되면 안되고, 메모리에 상주하고 있어야 한다. 이러한 메모리를 pinned memory라고 한다. pinned memory는 메모리에 항상 존재하는 것이 보장되므로 연산 속도도 더 빠르다.

    <p align="center"><img src="https://user-images.githubusercontent.com/84084372/198917869-a061cb7f-30b8-413c-86f9-e8bcabddbb6f.png"></p>

아래의 방법도 가능하다.

```python
host_mem = np.empty(size, dtype=dtype)
_, cuda_mem = cudart.cudaMallocAsync(host_mem.nbytes, stream)
```

<br>

### 3-3) async하게 GPU와 데이터 주고받기

#### 1) H2D

Host Input -> Device Input

```python
cudart.cudaMemcpyAsync(device_inputs[0], host_inputs[0].ctypes.data, host_inputs[0].nbytes,
                        cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, stream)
# stream 마지막 인자에 stream 생략 시, NULL Stream이 들어감
```

#### 2) 커널 연산
```
상황/모델에 따른 코드 작성
```

#### 3) D2H

Device Output -> Host Output
 
 ```python
 cudart.cudaMemcpyAsync(host_outputs[n].ctypes.data, cuda_outputs[n], host_outputs[n].nbytes,
                               cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, stream)
 ```

<br>

### 3-4) 모든 task 종료 대기

```
cudaDeviceSynchronize()
```

위에서 일을 다 넣어놓고, 모든 device의 일이 끝날 때까지 기다린다.<br>

또는 아래 코드와 같이 특정 stream만 기다릴 수도 있다.

```python
cudart.cudaStreamSynchronize(stream)
```

<br>
<br>

<div align='center'>
  <a href="https://on-jungwoan.github.io/internship/Itern-TIL-10/#1026">
    본문으로 돌아가기
  </a>
</div>  
