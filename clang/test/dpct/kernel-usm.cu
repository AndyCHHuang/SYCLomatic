// RUN: dpct --format-range=none -out-root %T %s --usm-level=restricted --cuda-include-path="%cuda-path/include" --sycl-named-lambda  -- -x cuda --cuda-host-only -std=c++14
// RUN: FileCheck %s --match-full-lines --input-file %T/kernel-usm.dp.cpp

#include <cuda_runtime.h>
#include <stdio.h>
#include <cassert>

// CHECK: void testDevice(const int *K) {
// CHECK-NEXT: int t = K[0];
// CHECK-NEXT: }
__device__ void testDevice(const int *K) {
  int t = K[0];
}

// CHECK: void testKernelPtr(const int *L, const int *M, int N, sycl::nd_item<3> item_ct1) {
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
  // CHECK: karg1 = sycl::malloc_device<int>(32 , dpct::get_default_queue());
  // CHECK-NEXT: karg2 = sycl::malloc_device<int>(32 , dpct::get_default_queue());
  cudaMalloc(&karg1, 32 * sizeof(int));
  cudaMalloc(&karg2, 32 * sizeof(int));

  int karg3 = 80;
  // CHECK:   dpct::get_default_queue().submit(
  // CHECK-NEXT:     [&](sycl::handler &cgh) {
  // CHECK-NEXT:       auto dpct_global_range = griddim * threaddim;
  // CHECK-EMPTY:
  // CHECK-NEXT:       cgh.parallel_for<dpct_kernel_name<class testKernelPtr_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:         sycl::nd_range<3>(sycl::range<3>(dpct_global_range.get(2), dpct_global_range.get(1), dpct_global_range.get(0)), sycl::range<3>(threaddim.get(2), threaddim.get(1), threaddim.get(0))),
  // CHECK-NEXT:         [=](sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:           testKernelPtr((const int *)karg1, karg2, karg3, item_ct1);
  // CHECK-NEXT:         });
  // CHECK-NEXT:     });
  testKernelPtr<<<griddim, threaddim>>>((const int *)karg1, karg2, karg3);
}

// CHECK:dpct::shared_memory<float, 1> result(32);
// CHECK-NEXT:void my_kernel(float* result, sycl::nd_item<3> item_ct1, float *resultInGroup) {
// CHECK-NEXT:  // __shared__ variable
// CHECK-NEXT:  resultInGroup[item_ct1.get_local_id(2)] = item_ct1.get_group(2);
// CHECK-NEXT:  memcpy(&result[item_ct1.get_group(2)*8], resultInGroup, sizeof(float)*8);
// CHECK-NEXT:}
// CHECK-NEXT:int run_foo5 () {
// CHECK-NEXT:  dpct::get_default_queue().submit(
// CHECK-NEXT:    [&](sycl::handler &cgh) {
// CHECK-NEXT:      sycl::accessor<float, 1, sycl::access::mode::read_write, sycl::access::target::local> resultInGroup_acc_ct1(sycl::range<1>(8), cgh);
// CHECK-EMPTY:
// CHECK-NEXT:      auto result_ct0 = result.get_ptr();
// CHECK-EMPTY:
// CHECK-NEXT:      cgh.parallel_for<dpct_kernel_name<class my_kernel_{{[0-9a-z]+}}>>(
// CHECK-NEXT:        sycl::nd_range<3>(sycl::range<3>(1, 1, 4) * sycl::range<3>(1, 1, 8), sycl::range<3>(1, 1, 8)),
// CHECK-NEXT:        [=](sycl::nd_item<3> item_ct1) {
// CHECK-NEXT:          my_kernel(result_ct0, item_ct1, resultInGroup_acc_ct1.get_pointer());
// CHECK-NEXT:        });
// CHECK-NEXT:    });
// CHECK-NEXT:  printf("%f ", result[10]);
// CHECK-NEXT:}
__managed__ __device__ float result[32];
__global__ void my_kernel(float* result) {
  __shared__ float resultInGroup[8]; // __shared__ variable
  resultInGroup[threadIdx.x] = blockIdx.x;
  memcpy(&result[blockIdx.x*8], resultInGroup, sizeof(float)*8);
}
int run_foo5 () {
  my_kernel<<<4, 8>>>(result);
  printf("%f ", result[10]);
}

// CHECK:dpct::shared_memory<float, 1> result2(32);
// CHECK-NEXT:int run_foo6 () {
// CHECK-NEXT:  dpct::get_default_queue().submit(
// CHECK-NEXT:    [&](sycl::handler &cgh) {
// CHECK-NEXT:      sycl::accessor<float, 1, sycl::access::mode::read_write, sycl::access::target::local> resultInGroup_acc_ct1(sycl::range<1>(8), cgh);
// CHECK-EMPTY:
// CHECK-NEXT:      auto result2_ct0 = result2.get_ptr();
// CHECK-EMPTY:
// CHECK-NEXT:      cgh.parallel_for<dpct_kernel_name<class my_kernel_{{[0-9a-z]+}}>>(
// CHECK-NEXT:        sycl::nd_range<3>(sycl::range<3>(1, 1, 4) * sycl::range<3>(1, 1, 8), sycl::range<3>(1, 1, 8)),
// CHECK-NEXT:        [=](sycl::nd_item<3> item_ct1) {
// CHECK-NEXT:          my_kernel(result2_ct0, item_ct1, resultInGroup_acc_ct1.get_pointer());
// CHECK-NEXT:        });
// CHECK-NEXT:    });
// CHECK-NEXT:  printf("%f ", result2[10]);
// CHECK-NEXT:}
__managed__ float result2[32];
int run_foo6 () {
  my_kernel<<<4, 8>>>(result2);
  printf("%f ", result2[10]);
}

// CHECK:dpct::shared_memory<float, 0> result3;
// CHECK-NEXT:int run_foo7 () {
// CHECK-NEXT:  dpct::get_default_queue().submit(
// CHECK-NEXT:    [&](sycl::handler &cgh) {
// CHECK-NEXT:      sycl::accessor<float, 1, sycl::access::mode::read_write, sycl::access::target::local> resultInGroup_acc_ct1(sycl::range<1>(8), cgh);
// CHECK-EMPTY:
// CHECK-NEXT:      auto result3_ct0 = result3.get_ptr();
// CHECK-EMPTY:
// CHECK-NEXT:      cgh.parallel_for<dpct_kernel_name<class my_kernel_{{[0-9a-z]+}}>>(
// CHECK-NEXT:        sycl::nd_range<3>(sycl::range<3>(1, 1, 4) * sycl::range<3>(1, 1, 8), sycl::range<3>(1, 1, 8)),
// CHECK-NEXT:        [=](sycl::nd_item<3> item_ct1) {
// CHECK-NEXT:          my_kernel(result3_ct0, item_ct1, resultInGroup_acc_ct1.get_pointer());
// CHECK-NEXT:        });
// CHECK-NEXT:    });
// CHECK-NEXT:  printf("%f ", result3[0]);
// CHECK-NEXT:}
__managed__ float result3;
int run_foo7 () {
  my_kernel<<<4, 8>>>(&result3);
  printf("%f ", result3);
}

// CHECK:dpct::shared_memory<float, 0> in;
// CHECK-NEXT:dpct::shared_memory<float, 0> out;
// CHECK-NEXT:void my_kernel2(float in, float *out, sycl::nd_item<3> item_ct1) {
// CHECK-NEXT:  if (item_ct1.get_local_id(2) == 0) {
// CHECK-NEXT:    memcpy(out, &in, sizeof(float));
// CHECK-NEXT:  }
// CHECK-NEXT:}
// CHECK-NEXT:int run_foo8() {
// CHECK-NEXT:  in[0] = 42;
// CHECK-NEXT:  dpct::get_default_queue().submit(
// CHECK-NEXT:    [&](sycl::handler &cgh) {
// CHECK-NEXT:      auto in_ct0 = in[0];
// CHECK-NEXT:      auto out_ct1 = out.get_ptr();
// CHECK-EMPTY:
// CHECK-NEXT:      cgh.parallel_for<dpct_kernel_name<class my_kernel2_{{[0-9a-z]+}}>>(
// CHECK-NEXT:        sycl::nd_range<3>(sycl::range<3>(1, 1, 4) * sycl::range<3>(1, 1, 8), sycl::range<3>(1, 1, 8)),
// CHECK-NEXT:        [=](sycl::nd_item<3> item_ct1) {
// CHECK-NEXT:          my_kernel2(in_ct0, out_ct1, item_ct1);
// CHECK-NEXT:        });
// CHECK-NEXT:    });
// CHECK-NEXT:  printf("%f ", out[0]);
// CHECK-NEXT:}
__managed__ float in;
__managed__ float out;
__global__ void my_kernel2(float in, float *out) {
  if (threadIdx.x == 0) {
    memcpy(out, &in, sizeof(float));
  }
}
int run_foo8() {
  in = 42;
  my_kernel2<<<4, 8>>>(in, &out);
  printf("%f ", out);
}
