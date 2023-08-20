---
title: "CUDA: Simple Kernel Assertion"
date: 2023-08-20
layout: post
---

Coming to debug CUDA kernels, many of you might have used `printf` to print out `blockIdx`, `threadIdx` to investigate what the threads are doing and what make them go wrong. This approach could get very messy sometimes as it might not be very informational when you debug on a large-sized kernel, which ends up spending tremendous time in debugging.

With `cassert`, the debug process could be much more effective when we are able to predict the expected behaviours of threads, such as offset boundaries, in our kernel codes. If a thread "violated" the assertion, its `blockIdx`, `threadIdx`, as well as the assertion would be printed out in the program. Here is the code example to reproduce a simple assertion":

```c++
#include <cassert>

__global__ void kernel() {
	assert(blockIdx.x < 3);
}

void trigger_assert() {
	kernel << < 6, 1 >> > ();
}

int main(void) {
	trigger_assert();
	return 0;
}

```
Output
```shell
C:\Users\seaso\source\repos\spoonbobo\babycuda\playground\assert\kernel.cu:9: block: [5,0,0], thread: [0,0,0] Assertion `blockIdx.x < 3` failed.
C:\Users\seaso\source\repos\spoonbobo\babycuda\playground\assert\kernel.cu:9: block: [4,0,0], thread: [0,0,0] Assertion `blockIdx.x < 3` failed.
C:\Users\seaso\source\repos\spoonbobo\babycuda\playground\assert\kernel.cu:9: block: [3,0,0], thread: [0,0,0] Assertion `blockIdx.x < 3` failed.
```
