---
layout: page_with_sidebar
title: "Use color image"
order: 11
---

## 개요

이번 강의는 **연습에 더 가까운** 내용입니다. 이전에 배운 다차원 스레드 인덱싱을 활용해서 컬러 이미지를 그레이스케일로 변환해보겠습니다.

## 컬러 이미지 vs 그레이스케일

- **그레이스케일**: 픽셀당 1개 값 (밝기만)
- **컬러 이미지**: 픽셀당 3개 값 (R, G, B)
32×32 컬러 이미지라면:

- 그레이스케일: 1,024개 값
- 컬러: 1,024 × 3 = 3,072개 값
## RGB to Grayscale 변환 공식

```bash
Gray = 0.299 × R + 0.587 × G + 0.114 × B
```


이 공식은 인간의 눈이 녹색에 가장 민감하고, 파란색에 가장 덜 민감하다는 점을 반영합니다.


![colorCat](/assets/images/use-color-image/colorcat.png)

## Code 예제

```c++
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION  
#include "stb_image_write.h"
#include <iostream>
#include <vector>

__global__ void colorToGrayscaleKernel(const uint8_t* colorInput, uint8_t* grayOutput)
{
    const int x = threadIdx.x;  // 0 ~ 31
    const int y = threadIdx.y;  // 0 ~ 31
    
    const int colorIdx = (y * 32 + x) * 3;

    const int grayIdx = y * 32 + x;

    const float r = static_cast<float>(colorInput[colorIdx + 0]);
    const float g = static_cast<float>(colorInput[colorIdx + 1]);
    const float b = static_cast<float>(colorInput[colorIdx + 2]);

    const float gray = 0.299f * r + 0.587f * g + 0.114f * b;

    grayOutput[grayIdx] = static_cast<uint8_t>(gray);
}

int main()
{
    // 32×32 컬러 이미지 로드 (3채널)
    int imgWidth, imgHeight, imgChannels;
    uint8_t* hostColorImage = stbi_load("cat32color.png",
        &imgWidth, &imgHeight, &imgChannels, 3); 
    
    assert(imgWidth == 32 && imgHeight == 32 && imgChannels == 3);
    
    uint8_t* deviceColorInput;
    uint8_t* deviceGrayOutput;
    cudaMalloc(&deviceColorInput, 32 * 32 * 3 * sizeof(uint8_t)); 
    cudaMalloc(&deviceGrayOutput, 32 * 32 * sizeof(uint8_t));     
    
    cudaMemcpy(deviceColorInput, hostColorImage, 32 * 32 * 3, cudaMemcpyHostToDevice);
    
    constexpr dim3 blockSize(32, 32); 
    colorToGrayscaleKernel<<<1, blockSize>>>(deviceColorInput, deviceGrayOutput);
    
    cudaDeviceSynchronize();
    
    // 흑백 결과를 CPU로 복사
    std::vector<uint8_t> hostGrayResult(32 * 32);
    cudaMemcpy(hostGrayResult.data(), deviceGrayOutput, 32 * 32, cudaMemcpyDeviceToHost);
    
    // 흑백 이미지 저장
    stbi_write_png("cat32gray_converted.png", 32, 32, 1, hostGrayResult.data(), 32);
    
    // 메모리 해제
    cudaFree(deviceColorInput);
    cudaFree(deviceGrayOutput);
    stbi_image_free(hostColorImage);
    
    std::cout << "Color to grayscale conversion completed!" << std::endl;
    return 0;
}
```

## 핵심 포인트

### 1. 메모리 레이아웃

**컬러 이미지 메모리 구조:**

```bash
[R0,G0,B0, R1,G1,B1, R2,G2,B2, ...]
```

### 2. 인덱스 계산

```c++
// 컬러: 픽셀당 3개 값이므로 × 3
const int colorIdx = (y * 32 + x) * 3;
// 그레이: 픽셀당 1개 값
const int grayIdx = y * 32 + x;
```

### 3.  픽셀 계산

각 스레드가 하나의 픽셀을 담당해서 RGB → Gray 변환을 수행합니다.


### 4. 결과확인


![grayCat](/assets/images/use-color-image/graycat.png)