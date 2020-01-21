// RUN: dpct --format-range=none -out-root %T %s --usm-level=restricted --cuda-include-path="%cuda-path/include" --sycl-named-lambda  -- -x cuda --cuda-host-only -std=c++14
// RUN: FileCheck %s --match-full-lines --input-file %T/kernel-usm.dp.cpp

#include <cuda_runtime.h>

#include <cassert>

// CHECK: void testDevice(const int *K) {
// CHECK-NEXT: int t = K[0];
// CHECK-NEXT: }
__device__ void testDevice(const int *K) {
  int t = K[0];
}

// CHECK: void testKernelPtr(const int *L, const int *M, int N,
// CHECK-NEXT: sycl::nd_item<3> item_ct1) {
// CHECK-NEXT: testDevice(L);
// CHECK-NEXT: int gtid = item_ct1.get_group(2) * item_ct1.get_local_range().get(2) + item_ct1.get_local_id(2);
// CHECK-NEXT: }
__global__ void testKernelPtr(const int *L, const int *M, int N) {
  testDevice(L);
  int gtid = blockIdx.x * blockDim.x + threadIdx.x;
}

int main() {
  dim3 griddim = 2;
  dim3 threaddim = 32;
  int *karg1, *karg2;
  // CHECK: karg1 = (int *)sycl::malloc_device(32 * sizeof(int), dpct::get_current_device(), dpct::get_default_context());
  cudaMalloc(&karg1, 32 * sizeof(int));
  // CHECK: karg2 = (int *)sycl::malloc_device(32 * sizeof(int), dpct::get_current_device(), dpct::get_default_context());
  cudaMalloc(&karg2, 32 * sizeof(int));

  int karg3 = 80;
  // CHECK:   dpct::get_default_queue_wait().submit(
  // CHECK-NEXT:     [&](sycl::handler &cgh) {
  // CHECK-NEXT:       auto dpct_global_range = griddim * threaddim;
  // CHECK-NEXT:       cgh.parallel_for<dpct_kernel_name<class testKernelPtr_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:         sycl::nd_range<3>(sycl::range<3>(dpct_global_range.get(2), dpct_global_range.get(1), dpct_global_range.get(0)), sycl::range<3>(threaddim.get(2), threaddim.get(1), threaddim.get(0))),
  // CHECK-NEXT:         [=](sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:           testKernelPtr((const int *)karg1, karg2, karg3, item_ct1);
  // CHECK-NEXT:         });
  // CHECK-NEXT:     });
  testKernelPtr<<<griddim, threaddim>>>((const int *)karg1, karg2, karg3);
}
