---
layout: page_with_sidebar
title: "modern CPP"
order: 7
---

## 들어가며

지금까지 우리는 C 스타일의 메모리 관리를 사용해왔습니다. `new[]`, `delete[]`, 그리고 raw pointer들을 직접 다루면서 CUDA의 기본 개념을 배웠습니다.

**왜 처음에 C 스타일을 사용했을까요?**

CUDA의 메모리 관리 API들(`cudaMalloc`, `cudaFree`, `cudaMemcpy`)이 모두 C 스타일의 포인터를 사용하기 때문입니다. Host 메모리도 같은 스타일로 관리하면 일관성이 있고, CUDA 메모리 관리의 개념을 더 명확하게 이해할 수 있습니다.

```c++
// CUDA 메모리 관리는 C 스타일
int32_t* deviceData;
cudaMalloc(&deviceData, size);// C 스타일 할당
cudaFree(deviceData);// C 스타일 해제// Host 메모리도 같은 스타일로 맞춤
int32_t* hostData = new int32_t[size];// 일관성 있는 스타일
delete[] hostData;
```

하지만 실제 프로젝트에서는 Modern C++ 스타일을 사용하는 것이 훨씬 안전하고 효율적입니다. 이번 시간에는 잠시 쉬어가면서 `std::vector`와 같은 Modern C++ 기능들을 CUDA와 함께 사용하는 방법을 알아보겠습니다.

## 왜 Modern C++를 사용해야 할까?

이제 CUDA의 메모리 관리 개념을 충분히 이해했으니, 더 안전하고 효율적인 Modern C++ 스타일로 전환할 때입니다.

### 기존 C 스타일의 문제점

```c++
// 위험한 C 스타일
int32_t *hostData = new int32_t[1024];
// ... 사용 ...
delete[] hostData;// 깜빡하면 메모리 누수!
```

### Modern C++ 스타일의 장점

```c++
// 안전한 Modern C++ 스타일
std::vector<int32_t> hostData(1024);
// 자동으로 메모리 해제됨!
```

**장점들:**

1. **자동 메모리 관리**: 메모리 누수 방지
1. **예외 안전성**: 예외가 발생해도 안전
1. **편리한 API**: `.size()`, `.data()`, `.push_back()` 등
1. **컴파일러 최적화**: 더 나은 성능 가능
1. **가독성**: 코드가 더 명확함
## Modern C++를 사용한 벡터 덧셈

이전 강의의 코드를 Modern C++ 스타일로 리팩토링해보겠습니다.

```c++
#include <iostream>
#include <vector>
#include <cuda_runtime.h>

__global__ void vectorAdd(const int32_t* dataA, const int32_t* dataB, int32_t* dataC)
{
    const int idx = threadIdx.x;
    dataC[idx] = dataA[idx] + dataB[idx];
}

int main()
{
    constexpr uint32_t dataLength = 1024;

    std::vector<int32_t> hostDataA(dataLength);
    std::vector<int32_t> hostDataB(dataLength);
    std::vector<int32_t> hostDataC(dataLength);

    for (size_t i = 0; i < dataLength; ++i)
    {
        hostDataA[i] = static_cast<int32_t>(i);
        hostDataB[i] = static_cast<int32_t>(i * 2);
        hostDataC[i] = 0;
    }

// Calculate memory size
    const size_t bytes = dataLength * sizeof(int32_t);

// Allocate device memory
    int32_t* deviceDataA = nullptr;
    int32_t* deviceDataB = nullptr;
    int32_t* deviceDataC = nullptr;

    cudaMalloc(&deviceDataA, bytes);
    cudaMalloc(&deviceDataB, bytes);
    cudaMalloc(&deviceDataC, bytes);

// Copy host to device using vector.data()
    cudaMemcpy(deviceDataA, hostDataA.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceDataB, hostDataB.data(), bytes, cudaMemcpyHostToDevice);

// Launch kernel
    vectorAdd<<<1, dataLength>>>(deviceDataA, deviceDataB, deviceDataC);

// Synchronize
    cudaDeviceSynchronize();

// Copy device to host
    cudaMemcpy(hostDataC.data(), deviceDataC, bytes, cudaMemcpyDeviceToHost);

// Print results using modern iteration
    std::cout << "First 10 results: ";
    for (size_t i = 0; i < 10; ++i)
    {
        std::cout << hostDataA[i] << "+" << hostDataB[i] << "=" << hostDataC[i] << " ";
    }
    std::cout << std::endl;

// Free device memory
    cudaFree(deviceDataA);
    cudaFree(deviceDataB);
    cudaFree(deviceDataC);

// Host memory automatically freed when vectors go out of scope!

    return 0;
}
```

## 주요 변경사항 분석

### 1. std::vector 사용

```c++
// Before: C style
int32_t *hostDataA = new int32_t[dataLength];

// After: Modern C++ style
std::vector<int32_t> hostDataA(dataLength);
```

**장점:**

- 자동 초기화 (모든 요소가 0으로 초기화됨)
- 자동 메모리 해제
- 크기 정보 포함 (`.size()` 메서드)
## 주의사항

### GPU 메모리는 여전히 수동 관리

```c++
// GPU memory still requires manual management
cudaMalloc(&deviceData, bytes);
// ... use ...
cudaFree(deviceData);// Must free manually!
```

CUDA API는 C 스타일이므로, GPU 메모리 관리는 여전히 수동으로 해야 합니다. 하지만 Host 메모리는 `std::vector`로 안전하게 관리할 수 있습니다.

## 정리

Modern C++를 CUDA와 함께 사용하면:

1. **안전성**: 메모리 누수 방지
1. **가독성**: 코드가 더 명확함
1. **유지보수성**: 버그 발생 가능성 감소
1. **생산성**: 더 빠른 개발 가능
**앞으로의 강의에서는 Modern C++ 스타일을 기본으로 사용하겠습니다.** 이제 메모리 관리 걱정 없이 CUDA의 핵심 개념들에 집중할 수 있습니다!

더 이상 `new[]`/`delete[]` 대신 `std::vector`를 사용하고, raw pointer보다는 안전한 메모리 관리 방식을 채택하겠습니다. CUDA의 메모리 관리 개념은 충분히 익혔으니, 이제 실용적이고 안전한 코드 작성에 집중할 때입니다.

## 다음 강의

다음 시간에는 Modern C++를 사용하여:

- **Grid와 Block 개념**
- **대용량 데이터 처리**
- **2차원 스레드 구조**
에 대해 배워보겠습니다.