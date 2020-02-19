// FIXME
// UNSUPPORTED: -windows-
// RUN: dpct --format-range=none --usm-level=none -out-root %T %s --cuda-include-path="%cuda-path/include" --sycl-named-lambda -- -x cuda --cuda-host-only
// RUN: FileCheck %s --match-full-lines --input-file %T/sharedmem_var_dynamic.dp.cpp

#include <stdio.h>
#define SIZE 100
// CHECK: void staticReverse(int *d, int n,
// CHECK-NEXT:  sycl::nd_item<3> item_ct1,
// CHECK-NEXT:  dpct::byte_t *dpct_local) {
// CHECK-NEXT:  auto s = (int *)dpct_local; // the size of s is dynamic
__global__ void staticReverse(int *d, int n) {
  extern __shared__ int s[]; // the size of s is dynamic
  int t = threadIdx.x;
  if (t < 64) {
    s[t] = d[t];
  }
}

// CHECK: template<typename TData>
// CHECK-NEXT: void templateReverse(TData *d, TData n,
// CHECK-NEXT: sycl::nd_item<3> item_ct1,
// CHECK-NEXT: dpct::byte_t *dpct_local) {
template<typename TData>
__global__ void templateReverse(TData *d, TData n) {

  // CHECK: auto s = (TData *)dpct_local; // the size of s is dynamic
  extern __shared__ TData s[]; // the size of s is dynamic
  int t = threadIdx.x;
  if (t < 64) {
    s[t] = d[t];
  }
}

template<typename T>
void testTemplate(){
  const int n = 64;
  T a[n], r[n], d[n];
  T *d_d;
  int mem_size = n * sizeof(T);
  cudaMalloc((void **)&d_d, mem_size);
  cudaMemcpy(d_d, a, mem_size, cudaMemcpyHostToDevice);

  // CHECK: {
  // CHECK-NEXT:   std::pair<dpct::buffer_t, size_t> d_d_buf_ct0 = dpct::get_buffer_and_offset(d_d);
  // CHECK-NEXT:   size_t d_d_offset_ct0 = d_d_buf_ct0.second;
  // CHECK-NEXT:   dpct::get_default_queue().submit(
  // CHECK-NEXT:     [&](sycl::handler &cgh) {
  // CHECK-NEXT:       sycl::accessor<dpct::byte_t, 1, sycl::access::mode::read_write, sycl::access::target::local> dpct_local_acc_ct1(sycl::range<1>(mem_size), cgh);
  // CHECK-NEXT:       auto d_d_acc_ct0 = d_d_buf_ct0.first.get_access<sycl::access::mode::read_write>(cgh);
  // CHECK-EMPTY:
  // CHECK-NEXT:       cgh.parallel_for<dpct_kernel_name<class templateReverse_{{[a-f0-9]+}}, T>>(
  // CHECK-NEXT:         sycl::nd_range<3>(sycl::range<3>(1, 1, 1) * sycl::range<3>(1, 1, n), sycl::range<3>(1, 1, n)),
  // CHECK-NEXT:         [=](sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:           T *d_d_ct0 = (T *)(&d_d_acc_ct0[0] + d_d_offset_ct0);
  // CHECK-NEXT:           templateReverse<T>(d_d_ct0, n, item_ct1, dpct_local_acc_ct1.get_pointer());
  // CHECK-NEXT:         });
  // CHECK-NEXT:     });
  // CHECK-NEXT: }
  templateReverse<T><<<1, n, mem_size>>>(d_d, n);
}

int main(void) {
  const int n = 64;
  int a[n], r[n], d[n];
  int *d_d;
  int mem_size = n * sizeof(int);
  cudaMalloc((void **)&d_d, mem_size);
  cudaMemcpy(d_d, a, mem_size, cudaMemcpyHostToDevice);
  // CHECK: {
  // CHECK-NEXT:   dpct::buffer_t d_d_buf_ct0 = dpct::get_buffer(d_d);
  // CHECK-NEXT:   dpct::get_default_queue().submit(
  // CHECK-NEXT:     [&](sycl::handler &cgh) {
  // CHECK-NEXT:       sycl::accessor<dpct::byte_t, 1, sycl::access::mode::read_write, sycl::access::target::local> dpct_local_acc_ct1(sycl::range<1>(mem_size), cgh);
  // CHECK-NEXT:       auto d_d_acc_ct0 = d_d_buf_ct0.get_access<sycl::access::mode::read_write>(cgh);
  // CHECK-EMPTY:
  // CHECK-NEXT:       cgh.parallel_for<dpct_kernel_name<class staticReverse_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:         sycl::nd_range<3>(sycl::range<3>(1, 1, 1) * sycl::range<3>(1, 1, n), sycl::range<3>(1, 1, n)),
  // CHECK-NEXT:         [=](sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:           staticReverse((int *)(&d_d_acc_ct0[0]), n, item_ct1, dpct_local_acc_ct1.get_pointer());
  // CHECK-NEXT:         });
  // CHECK-NEXT:     });
  // CHECK-NEXT: }
  staticReverse<<<1, n, mem_size>>>(d_d, n);
  cudaMemcpy(d, d_d, mem_size, cudaMemcpyDeviceToHost);

  // CHECK: {
  // CHECK-NEXT:   dpct::buffer_t d_d_buf_ct0 = dpct::get_buffer(d_d);
  // CHECK-NEXT:   dpct::get_default_queue().submit(
  // CHECK-NEXT:     [&](sycl::handler &cgh) {
  // CHECK-NEXT:       sycl::accessor<dpct::byte_t, 1, sycl::access::mode::read_write, sycl::access::target::local> dpct_local_acc_ct1(sycl::range<1>(sizeof(int)), cgh);
  // CHECK-NEXT:       auto d_d_acc_ct0 = d_d_buf_ct0.get_access<sycl::access::mode::read_write>(cgh);
  // CHECK-EMPTY:
  // CHECK-NEXT:       cgh.parallel_for<dpct_kernel_name<class staticReverse_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:         sycl::nd_range<3>(sycl::range<3>(1, 1, 1) * sycl::range<3>(1, 1, n), sycl::range<3>(1, 1, n)),
  // CHECK-NEXT:         [=](sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:           staticReverse((int *)(&d_d_acc_ct0[0]), n, item_ct1, dpct_local_acc_ct1.get_pointer());
  // CHECK-NEXT:         });
  // CHECK-NEXT:     });
  // CHECK-NEXT: }
  staticReverse<<<1, n, sizeof(int)>>>(d_d, n);

  // CHECK: {
  // CHECK-NEXT:   dpct::buffer_t d_d_buf_ct0 = dpct::get_buffer(d_d);
  // CHECK-NEXT:   dpct::get_default_queue().submit(
  // CHECK-NEXT:     [&](sycl::handler &cgh) {
  // CHECK-NEXT:       sycl::accessor<dpct::byte_t, 1, sycl::access::mode::read_write, sycl::access::target::local> dpct_local_acc_ct1(sycl::range<1>(4), cgh);
  // CHECK-NEXT:       auto d_d_acc_ct0 = d_d_buf_ct0.get_access<sycl::access::mode::read_write>(cgh);
  // CHECK-EMPTY:
  // CHECK-NEXT:       cgh.parallel_for<dpct_kernel_name<class templateReverse_{{[a-f0-9]+}}, int>>(
  // CHECK-NEXT:         sycl::nd_range<3>(sycl::range<3>(1, 1, 1) * sycl::range<3>(1, 1, n), sycl::range<3>(1, 1, n)),
  // CHECK-NEXT:         [=](sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:           templateReverse<int>((int *)(&d_d_acc_ct0[0]), n, item_ct1, dpct_local_acc_ct1.get_pointer());
  // CHECK-NEXT:         });
  // CHECK-NEXT:     });
  // CHECK-NEXT: }
  templateReverse<int><<<1, n, 4>>>(d_d, n);
}

