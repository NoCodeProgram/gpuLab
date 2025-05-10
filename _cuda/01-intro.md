---
layout: page_with_sidebar
title: "CUDA 소개"
order: 1
---
# Introduction

Tags: gpuLab

안녕하세요. GPU, CUDA 프로그래밍에 관심이 있으신 여러분 환영합니다.  GPU프로그램에 대해서 일반적으로 가지는 인식은

1. 연산속도를 빠르게 하기위해
2. 많은 양의 계산을 한번에 처리하기 위해

가 많을꺼에요.

이런 장점이 있음에도 불구하고 배우기 어렵다는 선입견 때문에 쉽게 접근하기 어려운것도 사실이에요. GPULab.kr 에서 제공하는 강의를 통해 누구나 관심이 있으신 분들은 아주 쉽게 GPU 프로그래밍을 배울수 있게 하는것이 목적이에요.

더 나아가 GPU 개발자가 많아짐으로서, 기회가 많아지고, 부가가치도 늘리며 , 고임금 일자리도 많아지게 하는게 목적이에요.

그래서 CPU가 GPU보다 얼마나 빠른데? 궁금하신 분들을 위해 CPU와 GPU의 갯수를 비교해 볼께요

공정함을 위해 데스크톱 시리즈를 비교할께요

AMD Ryzen시리즈 같은경우

![[https://www.amd.com/en/products/processors/desktops/ryzen.html#specifications](https://www.amd.com/en/products/processors/desktops/ryzen.html#specifications)](/assets/images/Introduction%201ee653e13b1280858685cda046b4cd44/image.png)

[https://www.amd.com/en/products/processors/desktops/ryzen.html#specifications](https://www.amd.com/en/products/processors/desktops/ryzen.html#specifications)

CPU core의 갯수가 16개에요

그럼 데스크톱 용도의 GPU의 코어를 확인해보면

![image.png](/assets/images/Introduction%201ee653e13b1280858685cda046b4cd44/image%201.png)

코어갯수가 21760개나 되요.

물론 코어의 갯수가 컴퓨터 연산속도를 그대로 반영하진 않지만 16개와 21760 라는 차이는 매우커요. 

아래그래프는 GPU와 CPU의 연산속도를 log scale로 나타낸거에요

![Code_Generated_Image.png](/assets/images/Introduction%201ee653e13b1280858685cda046b4cd44/Code_Generated_Image.png)

CPU가 발전을 하고 있지만, 단일 GPU로도 연산속도가 어림잡아 CPU보다 약 100배 정도 빠르다는것을 확인할수있어요

결국 쉽게말해서 

- CPU에서 1초 걸리는 연산이, GPU에서는 0.01초
- CPU 에서 1분 걸리는 연산이 , GPU 에서는 1초
- CPU 에서 1시간 걸리는 연산이, GPU 에서는 1분
- CPU 에서 하루 걸리는 연산이, GPU 에서는 20분

이렇게 단축된 시간으로, 새로운 산업들이 열릴수 있는거에요.  실시간 자율주행, medical image, simulation, financial industry 등의 연산이 가능한거죠. 그리고 여기에 있는 가치들이 여러분들이 CUDA를 배워야 하는 이유에요

그럼 이렇게 빠른 GPU를 활용하는 방법에는 AMD의 ROCm, 애플의 metal이 있고, 칩 제조사와 무관하게 활용하는 방법으로는 OpenCL이나 webGPU등이 있는데 왜 NVIDIA의 CUDA를 사용해야할까요?

당연하게도 오랜역사와 성공적으로 시장을 장악한 CUDA를 통해 배우는게 가장 쉽기 때문이에요. CUDA는 2008년에 처음 나왔고, 지속적인 버전업과 라이브러리 업데이트를 통해 많은 지원을 받을수 있어요.

그렇다고 해도 [GPULab.kr](http://GPULab.kr) 의 CUDA 강의는 GPU프로그래밍 알고리즘, 그리고 GPU 하드웨어에 더 초점을 맞출것이기 때문에, 나중에 다른 언어, 다른 칩을 사용한다고 해도 아주 쉽게 적응 할수 있을꺼에요.

그럼 CUDA를 통한 병렬프로그래밍 강의 바로 시작할께요.
