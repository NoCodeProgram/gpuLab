---
layout: page_with_sidebar
title: "CUDA 에러 처리"
order: 8
---

# 소개

프로그래밍을 할 때, 개발자가 코드를 잘못 작성했거나 API 자체에서 버그가 있는 경우, 에러는 피할 수 없습니다. CUDA API들 또한 에러를 반환합니다.
프로그램이 실행 중에 발생할 수 있는 다양한 오류 상황들을 적절히 처리하는 것은 안정적인 CUDA 애플리케이션 개발에 필수적입니다.


# CUDA Error Code 시스템

CUDA API 문서를 확인해보면, `cudaMalloc` API는 `cudaError_t` 타입을 반환합니다. `cudaMalloc`뿐만 아니라 `cudaMemcpy`, `cudaFree`와 같은 거의 모든 CUDA API는 에러 값을 반환합니다.

![cudamalloc](/assets/images/cuda-에러-처리/cudamalloc.png)


반환되는 cudaError_t를 더 살펴보면 여러 에러 코드들이 있습니다.

![cudaerror](/assets/images/cuda-에러-처리/cudaerror.png)

- **cudaSuccess**: API 호출이 성공적으로 완료된 경우 반환됩니다
- **기타 에러 코드들**: API 호출이 실패한 경우 다양한 에러 코드들이 반환됩니다
이러한 값들을 이용하여 CUDA 에러를 적절히 처리할 수 있습니다.


이를 이용하여 에러를 체크하는 함수를 작성해 보겠습니다. 


## C++20을 이용한 에러 체크 함수

많은 CUDA 예제 코드에서는 매크로를 사용한 에러 처리를 볼 수 있습니다. 하지만 C++20부터는 매크로를 사용하지 않고도 효과적인 에러 체크 함수를 작성할 수 있습니다:

```c++
#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <source_location>

inline void cudaCheckErr(cudaError_t err, const std::source_location& loc = std::source_location::current())
{
    if (err != cudaSuccess) {
        std::cerr << "CUDA error at " << loc.file_name() << ":" << loc.line() 
                  << " - " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}
```

###  완전한 예제 코드

```c++
#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <source_location>

inline void cudaCheckErr(cudaError_t err, const std::source_location& loc = std::source_location::current())
{
    if (err != cudaSuccess) {
        std::cerr << "CUDA error at " << loc.file_name() << ":" << loc.line() 
                  << " - " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

__global__ void add100(int32_t* data)
{
    const int idx = threadIdx.x;
    data[idx] = data[idx] + 100;
}

int main()
{
    constexpr uint32_t dataLength = 1024;
    std::vector<int32_t> hostData(dataLength);
    for (int32_t i = 0; i < dataLength; ++i)
    {
        hostData[i] = i; 
    }

    int32_t* deviceData;
    const auto mallocErr = cudaMalloc(&deviceData, dataLength * sizeof(int32_t));
    cudaCheckErr(mallocErr);

    cudaCheckErr(cudaMemcpy(deviceData, hostData.data(), dataLength * sizeof(int32_t), cudaMemcpyHostToDevice));
    add100 <<<1, dataLength >>> (deviceData);
    const cudaError_t launchErr = cudaGetLastError();    
    cudaCheckErr(launchErr);

    cudaCheckErr(cudaDeviceSynchronize());

    cudaCheckErr(cudaMemcpy(hostData.data(), deviceData, dataLength * sizeof(int32_t), cudaMemcpyDeviceToHost));
    cudaFree(deviceData);

    for (int32_t i = 0; i < 10; ++i)
    {
        std::cout << hostData[i] << " ";
    }    
    return 0;
}
```


### 커널 실행 에러 처리

### cudaGetLastError() 함수

커널을 실행할 때는 다른 CUDA API들과 달리 에러 값을 직접 반환하지 않습니다. 이런 경우에는 `cudaGetLastError()`를 통해 런타임에서 마지막에 발생한 에러를 받아올 수 있습니다.

```c++
add100 <<<1, dataLength >>> (deviceData);
const cudaError_t launchErr = cudaGetLastError();
cudaCheckErr(launchErr);
```


위 코드를 빌드, 실행하면, 프로그램이 정상적으로 동작합니다


```c++
#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <source_location>

inline void cudaCheckErr(cudaError_t err, const std::source_location& loc = std::source_location::current())
{
    if (err != cudaSuccess) {
        std::cerr << "CUDA error at " << loc.file_name() << ":" << loc.line() 
                  << " - " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

__global__ void add100(int32_t* data)
{
    const int idx = threadIdx.x;
    data[idx] = data[idx] + 100;
}

int main()
{
    constexpr uint32_t dataLength = 1025;
    std::vector<int32_t> hostData(dataLength);
    for (int32_t i = 0; i < dataLength; ++i)
    {
        hostData[i] = i; 
    }

    int32_t* deviceData;
    const auto mallocErr = cudaMalloc(&deviceData, dataLength * sizeof(int32_t));
    cudaCheckErr(mallocErr);

    cudaCheckErr(cudaMemcpy(deviceData, hostData.data(), dataLength * sizeof(int32_t), cudaMemcpyHostToDevice));
    add100 <<<1, dataLength >>> (deviceData);
    const cudaError_t launchErr = cudaGetLastError();    
    cudaCheckErr(launchErr);

    cudaCheckErr(cudaDeviceSynchronize());

    cudaCheckErr(cudaMemcpy(hostData.data(), deviceData, dataLength * sizeof(int32_t), cudaMemcpyDeviceToHost));
    cudaFree(deviceData);

    for (int32_t i = 0; i < 10; ++i)
    {
        std::cout << hostData[i] << " ";
    }    
    return 0;
}
```

하지만 dataLength 를 1025 개로 올려, block당 thread의 갯수를 1025개로 만든다면, 에러가 납니다. 이는 CUDA 디바이스의 하드웨어 제한 때문입니다. 만약 에러 체크를 하지 않고, 빌드, 실행을 했다면 이 문제를 찾는데 아주 많은 시간이 걸렸을 겁니다.


### 에러 체크의 성능 영향

`cudaCheckErr`와 같은 에러 체크 함수는 성능에 크게 영향을 미치지 않습니다. 따라서 Release 빌드와 Debug 빌드 모두에서 활성화시키는 것을 권장합니다.

### 실제 개발에서의 적용

에러 처리는 매우 중요하지만, 학습 과정에서는 코드의 가독성을 위해 에러 체크 코드를 생략할 것입니다. 하지만 실제 프로덕션 코드에서는 반드시 적절한 에러 처리를 포함해야 합니다.