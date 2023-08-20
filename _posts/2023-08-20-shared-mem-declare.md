---
title: "CUDA: Shared Memory Declaration"
date: 2023-08-20
layout: post
---

[Shared memory](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#device-memory-accesses) is one of [Device Memory Access (DMA)](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#device-memory-accesses) in CUDA paradigm. It provides a way for us to load data in global memory in cache on chip, enabling faster data access (around 100X compard to global memory).

There are 2 ways to declare and use shared memory in kernels: *static shared memory* (pre-determined) and *dynamic shared memory* (determined in runtime). 

Static shared memory
```c++
template<typename T>
__global__ void kernel(T* src) {
    __shared__ T smem[32];
    // do something with shared memory
}
```
Since it's static, there is no need to configure shared memory allocated for `kernel` in [kernel configurations](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#execution-configuration).

Dynamic shared memory
```c++
template<typename T>
__global__ void kernel(T* src) {
    extern __shared__ T smem[];
    // do something with shared memory
}

// kernel execution configuration
kernel<<<grid, block, smem_size>>>(src);
```
To use dynamic shared memory, we have to explicitly specify what size of `smem_size` we want to use for `kernel`.


Further readings:
* [using-shared-memory-cuda-cc](https://developer.nvidia.com/blog/using-shared-memory-cuda-cc/)