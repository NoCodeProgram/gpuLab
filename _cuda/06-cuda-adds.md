---
layout: page_with_sidebar
title: "CUDA Adds"
order: 6
---

## 들어가며

이전 시간에는 128개의 element 에 100을 더하는 아주 간단한 CUDA 프로그램을 만들어 보았습니다. 이번에는 cuda programming pattern에 더 익숙해지기 위해, 벡터 덧셈(Vector Addition)을 해보겠습니다.

벡터 덧셈은 GPU 프로그래밍에서 가장 기본적이면서도 중요한 연산 중 하나입니다. 두 개의 배열 A와 B가 있을 때, A[i] + B[i] = C[i]를 모든 요소에 대해 병렬로 수행하는 것입니다.

이번에는 스레드 수도 1024개로 늘려서 CUDA의 진정한 병렬 처리 능력을 느껴보겠습니다.

## 벡터 덧셈 CUDA 예제

```c++

#include <iostream>
#include <cuda_runtime.h>

__global__ void vectorAdd(const int32_t* dataA, const int32_t* dataB, int32_t* dataC)
{
    const int idx = threadIdx.x;
    dataC[idx] = dataA[idx] + dataB[idx];
}

void cpuVectorAdd(const int32_t* dataA, const int32_t* dataB, int32_t* dataC, const int size)
{
    for(int32_t idx = 0; idx < size; ++idx)
    {
        dataC[idx] = dataA[idx] + dataB[idx];
    }
}

int main()
{
    constexpr uint32_t dataLength = 1024;

// Allocate host memory
    int32_t *hostDataA = new int32_t[dataLength];
    int32_t *hostDataB = new int32_t[dataLength];
    int32_t *hostDataC = new int32_t[dataLength];

// Initialize data
    for (int32_t i = 0; i < dataLength; ++i)
    {
        hostDataA[i] = i;// A = [0, 1, 2, 3, ...]
        hostDataB[i] = i * 2;// B = [0, 2, 4, 6, ...]
        hostDataC[i] = 0;// C = [0, 0, 0, 0, ...]
    }

// Allocate device memory
    int32_t* deviceDataA = nullptr;
    int32_t* deviceDataB = nullptr;
    int32_t* deviceDataC = nullptr;

    cudaMalloc(&deviceDataA, dataLength * sizeof(int32_t));
    cudaMalloc(&deviceDataB, dataLength * sizeof(int32_t));
    cudaMalloc(&deviceDataC, dataLength * sizeof(int32_t));

// Copy host to device memory
    cudaMemcpy(deviceDataA, hostDataA, dataLength * sizeof(int32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceDataB, hostDataB, dataLength * sizeof(int32_t), cudaMemcpyHostToDevice);

// Launch kernel
    vectorAdd <<<1, dataLength >>> (deviceDataA, deviceDataB, deviceDataC);

// Synchronize
    cudaDeviceSynchronize();

// Copy device to host memory
    cudaMemcpy(hostDataC, deviceDataC, dataLength * sizeof(int32_t), cudaMemcpyDeviceToHost);

// Print results (first 10 and last 10 elements)
    std::cout << "First 10 : ";
    for (int32_t i = 0; i < 10; ++i)
    {
        std::cout << hostDataC[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "Last 10 : ";
    for (int32_t i = dataLength - 10; i < static_cast<int32_t>(dataLength); ++i)
    {
        std::cout << hostDataC[i] << " ";
    }
    std::cout << std::endl;

// Free memory
    cudaFree(deviceDataA);
    cudaFree(deviceDataB);
    cudaFree(deviceDataC);

    delete[] hostDataA;
    delete[] hostDataB;
    delete[] hostDataC;

    return 0;
}

```

## 코드 분석

### 1. 세 개의 Host 메모리 할당

이번에는 세 개의 배열이 필요합니다.

```c++
constexpr uint32_t dataLength = 1024;

int32_t *hostDataA = new int32_t[dataLength];
int32_t *hostDataB = new int32_t[dataLength];
int32_t *hostDataC = new int32_t[dataLength];
```

- `hostDataA`: 첫 번째 입력 벡터
- `hostDataB`: 두 번째 입력 벡터
- `hostDataC`: 결과를 저장할 벡터
### 2. 데이터 초기화

```c++
for (int32_t i = 0; i < dataLength; ++i)
{
    hostDataA[i] = i;// A = [0, 1, 2, 3, ...]
    hostDataB[i] = i * 2;// B = [0, 2, 4, 6, ...]
    hostDataC[i] = 0;// C = [0, 0, 0, 0, ...]
}
```

테스트를 위해 간단한 패턴으로 데이터를 초기화합니다:

- A 배열: 0, 1, 2, 3, 4, ...
- B 배열: 0, 2, 4, 6, 8, ...
- 결과적으로 C 배열: 0, 3, 6, 9, 12, ... (A[i] + B[i])
### 3. 세 개의 Device 메모리 할당

```c++
int32_t* deviceDataA;
int32_t* deviceDataB;
int32_t* deviceDataC;

cudaMalloc(&deviceDataA, dataLength * sizeof(int32_t));
cudaMalloc(&deviceDataB, dataLength * sizeof(int32_t));
cudaMalloc(&deviceDataC, dataLength * sizeof(int32_t))
```

GPU 메모리에도 세 개의 공간이 필요합니다.

### 4. 입력 데이터만 GPU로 복사

```c++
cudaMemcpy(deviceDataA, hostDataA, dataLength * sizeof(int32_t), cudaMemcpyHostToDevice);
cudaMemcpy(deviceDataB, hostDataB, dataLength * sizeof(int32_t), cudaMemcpyHostToDevice);
```

주목할 점은 `deviceDataC`는 복사하지 않는다는 것입니다. 왜냐하면 결과를 저장할 공간이므로, 초기값이 필요 없기 때문입니다.

### 5. 1024개 스레드로 커널 실행

```c++
vectorAdd <<<1, dataLength >>> (deviceDataA, deviceDataB, deviceDataC);

__global__ void vectorAdd(int32_t* dataA, int32_t* dataB, int32_t* dataC)
{
    const int idx = threadIdx.x;
    dataC[idx] = dataA[idx] + dataB[idx];
}
```

이번에는 **1024개의 스레드**가 동시에 실행됩니다!

- Thread 0: `dataC[0] = dataA[0] + dataB[0]`
- Thread 1: `dataC[1] = dataA[1] + dataB[1]`
- Thread 2: `dataC[2] = dataA[2] + dataB[2]`
- ...
- Thread 1023: `dataC[1023] = dataA[1023] + dataB[1023]`
CPU에서 1024번의 for loop를 도는 대신, GPU에서는 1024개 스레드가 **동시에** 연산을 수행합니다.

### CPU vs GPU 비교

CPU 버전이라면 이렇게 작성해야 합니다:

```c++
void cpuVectorAdd(int32_t* dataA, int32_t* dataB, int32_t* dataC, int size)
{
    for(int32_t idx = 0; idx < size; ++idx)
    {
        dataC[idx] = dataA[idx] + dataB[idx];
    }
}
```

- **CPU**: 1024번의 순차적 실행
- **GPU**: 1024개 스레드의 병렬 실행
## CUDA 스레드의 한계

CUDA에서 한 블록당 최대 스레드 수는 **1024개**입니다. 이는 GPU 하드웨어 및 CUDA 제약 사항 입니다. 만약 1024개보다 많은 요소를 처리하려면 한개의 block이 아닌, 여러개의 block을 다뤄야 합니다. 다만 이 부분은 하드웨어에 대한 이해도 필요하기 때문에, 조금 더 천천히 배우도록 하겠습니다.

## 다음 강의

지금까지 아주 기초적인 CUDA call을 다뤄봤습니다. block을 배우기 전에, cuda에서 multi dimension을 다루는 법에 대해 배우도록 하겠습니다