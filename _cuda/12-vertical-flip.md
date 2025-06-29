---
layout: page_with_sidebar
title: "vertical Flip"
order: 12
---

# CUDA 이미지 상하반전 연습

## 개요

이번에도 **연습 강의**입니다! 이전에 배운 색상 반전(invert)과 비슷하지만, 이번에는 **상하반전(Vertical Flip)**을 해보겠습니다.

## 색상 반전 vs 상하반전

- **색상 반전**: 픽셀 값 자체를 변경 (255 - pixel)
- **상하반전**: 픽셀의 **위치**를 변경 (위아래 뒤바꿈)
## 상하반전 원리

```bash
원본:          상하반전 후:
[0행] aaaa  →  [0행] dddd
[1행] bbbb  →  [1행] cccc
[2행] cccc  →  [2행] bbbb
[3행] dddd  →  [3행] aaaa
```

**핵심**: `y`번째 행이 `(height-1-y)`번째 행과 바뀝니다!

## 완전한 코드

```c++
#include <iostream>
#include <cuda_runtime.h>
#include <vector>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION  
#include "stb_image_write.h"

__global__ void verticalFlipKernel(const uint8_t* input, uint8_t* output)
{
    const int32_t x = threadIdx.x;  // 0 ~ 31
    const int32_t y = threadIdx.y;  // 0 ~ 31
    const int32_t targetY = 31 - y;

    const int32_t inputIdx = y * 32 + x;
    const int32_t outputIdx = targetY * 32 + x;
    
    output[outputIdx] = input[inputIdx];
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
    
    // Allocate GPU memory (for input and output)
    uint8_t* deviceInputPtr;
    uint8_t* deviceOutputPtr;
    cudaMalloc(&deviceInputPtr, imgBytes);
    cudaMalloc(&deviceOutputPtr, imgBytes);

    // Copy original image to GPU
    cudaMemcpy(deviceInputPtr, hostImage, imgBytes, cudaMemcpyHostToDevice);

    // Launch kernel
    constexpr dim3 blockSize(32, 32);
    verticalFlipKernel<<<1, blockSize>>>(deviceInputPtr, deviceOutputPtr);
    cudaDeviceSynchronize();
    
    // Copy result back to CPU
    cudaMemcpy(hostImage, deviceOutputPtr, imgBytes, cudaMemcpyDeviceToHost);

    // Save vertically flipped image
    stbi_write_png("flipped_cat32gray.png", imgWidth, imgHeight, 
                   imgChannels, hostImage, imgWidth * sizeof(uint8_t));

    // Free memory
    cudaFree(deviceInputPtr);
    cudaFree(deviceOutputPtr);
    stbi_image_free(hostImage);
    
    std::cout << "Vertical flip completed!" << std::endl;
    return 0;
}
```

## 핵심 포인트

### 1. 좌표 변환 공식

```c++
const int32_t targetY = 31 - y;
const int32_t outputIdx = targetY * 32 + x;
```

- `y = 0` (맨 위) → `targetY = 31` (맨 아래)
- `y = 31` (맨 아래) → `targetY = 0` (맨 위)
### 2. 입력과 출력 분리

이전 색상 반전과 달리 **별도의 출력 배열**이 필요합니다:

```c++
// 색상 반전: 제자리에서 수정
imgPtr[idx] = 255 - imgPtr[idx];

// 상하반전: 다른 위치로 복사
output[outputIdx] = input[inputIdx];
```

**중요**: 만약 같은 배열을 입력과 출력으로 사용하면 **데이터 덮어쓰기 문제**가 발생합니다!

예를 들어, 스레드 A가 위쪽 픽셀을 아래쪽으로 복사하는 동시에 스레드 B가 아래쪽 픽셀을 위쪽으로 복사하면, 원본 데이터가 복사되기 전에 덮어써져서 원본 값이 손실될 수 있습니다.


### 결과


![inImage](/assets/images/vertical-flip/inimage.png)


![outImage](/assets/images/vertical-flip/outimage.png)

## 다음 강의 예고

**드디어 Block에 대해 배워보겠습니다!**

지금까지는 32×32 = 1,024개 스레드가 한계였지만, 이제 여러 블록을 사용해서 더 큰 이미지도 처리할 수 있게 됩니다.