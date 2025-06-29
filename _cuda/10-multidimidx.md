---
layout: page_with_sidebar
title: "multiDimIdx"
order: 10
---

# Intro

이번 강의에서는 **Multi-Dimensional Thread Index**에 대해 알아보겠습니다.

지금까지는 1차원 배열에서 각 스레드가 자신의 고유 번호를 얻기 위해 `threadIdx.x`를 사용해왔습니다:

```c++
__global__ void add100(int32_t* data)
{
    const int idx = threadIdx.x;
    data[idx] = data[idx] + 100;
}
```


하지만 실제로 CUDA를 사용하는 주된 이유는 **데이터 집약적(Data Intensive)** 연산을 빠르게 처리하기 위함입니다.


## 다차원 데이터의 필요성

### 대표적인 다차원 데이터

- **이미지**: `width × height` (2차원)
- **영상**: `width × height × time` (3차원)
- **의료 데이터**: CT, MRI 등 `x × y × z` (3차원)

### 다차원 인덱싱의 장점

다차원 데이터에 GPU 스레드가 접근할 때, 각 스레드를 2D 또는 3D로 인덱싱할 수 있다면 프로그래밍이 훨씬 직관적이고 쉬워집니다.

## 실습 예제: 이미지 반전

### 예제 준비

- **이미지 크기**: 32×32 픽셀 (총 1,024개 픽셀)
- **형식**: 그레이스케일
- **스레드 구성**: 1개 블록, 32×32 스레드 배치
> 한 블록당 최대 스레드 수는 1,024개이므로, 32×32 = 1,024개의 스레드를 사용할 수 있습니다.

![catGray32](/assets/images/multidimidx/catgray32.png)

### stb image library

해당 예제를 만들기 위해서는, 이미지를 읽고 쓰는 함수가 필요합니다. standard C++에서는 이미지 읽기 쓰기에 대한 라이브러리를 제공하지 않기 때문에, 3rd party를 사용해야합니다. 라이센스가 free인 여러 open source들 중에서 


에서 제공하는 

- image loader: stb_image.h
- image writer: stb_image_write.h
를 사용해서 이미지를 읽고 씁니다.   보다 시피 public 도메인이고, header only library 이기 때문에 import , 컴파일이 쉽습니다.


### Code

```c++
#include <iostream>
#include <cuda_runtime.h>
#include <vector>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION  
#include "stb_image_write.h"

__global__ void invertKernel(uint8_t* imgPtr)
{
    int32_t x = threadIdx.x;
    int32_t y = threadIdx.y;
    int32_t idx = y * 32 + x;

    imgPtr[idx] = 255 - imgPtr[idx];
}

int main()
{
    // Load image
    int imgWidth, imgHeight, imgChannels;
    uint8_t* hostImage = stbi_load("cat32gray.png", 
                                   &imgWidth, &imgHeight, &imgChannels, 1);

    assert(imgWidth == 32 && imgHeight == 32 && imgChannels == 1);
    
    constexpr int32_t imgSize = 32 * 32;
    constexpr size_t imgBytes = imgSize * sizeof(uint8_t);
    
    uint8_t* deviceImgPtr;
    cudaMalloc(&deviceImgPtr, imgBytes);

    cudaMemcpy(deviceImgPtr, hostImage, imgBytes, cudaMemcpyHostToDevice);

    constexpr dim3 blockSize(32, 32);
    invertKernel<<<1, blockSize>>>(deviceImgPtr);
    cudaDeviceSynchronize();
    
    cudaMemcpy(hostImage, deviceImgPtr, imgBytes, cudaMemcpyDeviceToHost);

    stbi_write_png("inverted_cat32gray.png", imgWidth, imgHeight, imgChannels, hostImage, imgWidth * sizeof(uint8_t));

    cudaFree(deviceImgPtr);
    stbi_image_free(hostImage);
    
    std::cout << "Image inversion completed!" << std::endl;
    return 0;
}
```

### 이미지 읽기

이미지는 2차원으로 보이지만, 실제 메모리에서는 **1차원 배열**로 연속적으로 저장됩니다:

![catgray](/assets/images/multidimidx/catgray.png)



### dim3 타입

`dim3`는 CUDA에서 제공하는 3차원 정수 타입입니다:

```c++
dim3 blockSize(32, 32);// 32×32×1 (z는 기본값 1)
```

### 2차원 스레드 인덱싱

```c++
__global__ void kernel(uint8_t* data)
{
    int x = threadIdx.x;// 0~31
    int y = threadIdx.y;// 0~31

// 각 스레드가 고유한 (x,y) 좌표를 가짐
    int idx = y * 32 + x;

// 해당 픽셀 처리
    data[idx] = 255 - data[idx];
}
```

### 5.3 커널 실행 과정

1. **1,024개의 스레드**가 동시에 실행됩니다
1. 각 스레드는 **하나의 픽셀**을 담당합니다
1. 스레드 `(x,y)`는 이미지의 `(x,y)` 위치 픽셀을 처리합니다
1. 모든 스레드가 **병렬로** 동작하여 전체 이미지를 처리합니다
## 결과확인

![colorInversion](/assets/images/multidimidx/colorinversion.png)