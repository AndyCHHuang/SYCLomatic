// RUN: syclct -out-root %T %s -- -x cuda --cuda-host-only --cuda-path=%cuda-path
// RUN: FileCheck %s --match-full-lines --input-file %T/max_min_windows.sycl.cpp

#if defined(_WIN32) || defined(WIN32)
#include <Windows.h>
#endif

__global__ void test_max_min(void) {
  float a = 2.0, b = 3.0;

  // CHECK: float c = cl::sycl::max(a, b);
  float c = max(a, b);

  // CHECK: float d = cl::sycl::min(a, b);
  float d = min(a, b);
}
