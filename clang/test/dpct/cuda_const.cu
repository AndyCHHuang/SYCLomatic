// RUN: dpct --format-range=none --usm-level=none -out-root %T %s --cuda-include-path="%cuda-path/include" --sycl-named-lambda -- -x cuda --cuda-host-only
// RUN: FileCheck %s --match-full-lines --input-file %T/cuda_const.dp.cpp

#include <stdio.h>

#define NUM_ELEMENTS 16
const unsigned num_elements = 16;

class TestStruct {
public:
  __device__ void test() {}
};

// CHECK: dpct::constant_memory<TestStruct, 0> t1;
__constant__ TestStruct t1;

// CHECK: void member_acc(TestStruct t1) {
// CHECK-NEXT:  t1.test();
// CHECK-NEXT:}
__global__ void member_acc() {
  t1.test();
}
// CHECK: dpct::constant_memory<float, 1> const_angle(360);
// CHECK: dpct::constant_memory<float, 2> const_float(NUM_ELEMENTS, num_elements * 2);
__constant__ float const_angle[360], const_float[NUM_ELEMENTS][num_elements * 2];
// CHECK: dpct::constant_memory<sycl::double2, 0> vec_d;
__constant__ double2 vec_d;

// CHECK: dpct::device_memory<int, 1> const_ptr;
__constant__ int *const_ptr;

// CHECK:void simple_kernel(float *d_array, sycl::nd_item<3> [[ITEM:item_ct1]],
// CHECK-NEXT:              float *const_angle, int *const_ptr) {
// CHECK-NEXT:  int index;
// CHECK-NEXT:  index = [[ITEM]].get_group(2) * [[ITEM]].get_local_range().get(2) + [[ITEM]].get_local_id(2);
// CHECK-NEXT:  const_ptr[index] = index;
// CHECK-NEXT:  if (index < 360) {
// CHECK-NEXT:    d_array[index] = const_angle[index];
// CHECK-NEXT:  }
// CHECK-NEXT:  return;
// CHECK-NEXT:}
__global__ void simple_kernel(float *d_array) {
  int index;
  index = blockIdx.x * blockDim.x + threadIdx.x;
  const_ptr[index] = index;
  if (index < 360) {
    d_array[index] = const_angle[index];
  }
  return;
}

// CHECK: dpct::constant_memory<float, 0> const_one;
__constant__ float const_one;

// CHECK:void simple_kernel_one(float *d_array, sycl::nd_item<3> [[ITEM:item_ct1]],
// CHECK-NEXT:                  dpct::accessor<float, dpct::constant, 2> const_float,
// CHECK-NEXT:                  float const_one) {
// CHECK-NEXT:  int index;
// CHECK-NEXT:  index = [[ITEM]].get_group(2) * [[ITEM]].get_local_range().get(2) + [[ITEM]].get_local_id(2);
// CHECK-NEXT:  if (index < 33) {
// CHECK-NEXT:    d_array[index] = const_one + const_float[index][index];
// CHECK-NEXT:  }
// CHECK-NEXT:  return;
// CHECK-NEXT:}
__global__ void simple_kernel_one(float *d_array) {
  int index;
  index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < 33) {
    d_array[index] = const_one + const_float[index][index];
  }
  return;
}

int main(int argc, char **argv) {
  int size = 3200;
  int *d_int;
  float *d_array;
  float h_array[360];

  // CHECK: dpct::dpct_malloc((void **)&d_array, sizeof(float) * size);
  cudaMalloc((void **)&d_array, sizeof(float) * size);
  // CHECK: dpct::dpct_malloc(&d_int, sizeof(int) * size);
  cudaMalloc(&d_int, sizeof(int) * size);

  // CHECK: dpct::dpct_memset(d_array, 0, sizeof(float) * size);
  cudaMemset(d_array, 0, sizeof(float) * size);

  for (int loop = 0; loop < 360; loop++)
    h_array[loop] = acos(-1.0f) * loop / 180.0f;

  // CHECK:   const_ptr.assign(d_int, sizeof(int) * size);
  cudaMemcpyToSymbol(const_ptr, &d_int, sizeof(int *));
  // CHECK:/*
  // CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT:*/
  // CHECK-NEXT:   (dpct::dpct_memcpy(const_angle.get_ptr(), &h_array[0], sizeof(float) * 360), 0);
  cudaMemcpyToSymbol(&const_angle[0], &h_array[0], sizeof(float) * 360);

  // CHECK:/*
  // CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT:*/
  // CHECK-NEXT:   (dpct::dpct_memcpy(const_angle.get_ptr() + sizeof(float) * (3), &h_array[0], sizeof(float) * 357), 0);
  cudaMemcpyToSymbol(&const_angle[3], &h_array[0], sizeof(float) * 357);

  // CHECK:  /*
  // CHECK-NEXT:  DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT:  */
  // CHECK-NEXT:  (dpct::dpct_memcpy(&h_array[0], const_angle.get_ptr() + sizeof(float) * (3), sizeof(float) * 357), 0);
  cudaMemcpyFromSymbol(&h_array[0], &const_angle[3], sizeof(float) * 357);

  #define NUM 3
  // CHECK:/*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: (dpct::dpct_memcpy(const_angle.get_ptr() + sizeof(float) * (3+NUM), &h_array[0], sizeof(float) * 354), 0);
  cudaMemcpyToSymbol(&const_angle[3+NUM], &h_array[0], sizeof(float) * 354);

  // CHECK:  /*
  // CHECK-NEXT:  DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT:  */
  // CHECK-NEXT:  (dpct::dpct_memcpy(&h_array[0], const_angle.get_ptr() + sizeof(float) * (3+NUM), sizeof(float) * 354), 0);
  cudaMemcpyFromSymbol(&h_array[0], &const_angle[3+NUM], sizeof(float) * 354);

  // CHECK:   dpct::get_default_queue().submit(
  // CHECK-NEXT:     [&](sycl::handler &cgh) {
  // CHECK-NEXT:       auto t1_acc_ct1 = t1.get_access(cgh);
  // CHECK-EMPTY:
  // CHECK-NEXT:       cgh.parallel_for<dpct_kernel_name<class member_acc_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:         sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
  // CHECK-NEXT:         [=](sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:           member_acc(t1_acc_ct1);
  // CHECK-NEXT:         });
  // CHECK-NEXT:     });
  member_acc<<<1, 1>>>();
  // CHECK: {
  // CHECK-NEXT:   dpct::buffer_t d_array_buf_ct0 = dpct::get_buffer(d_array);
  // CHECK-NEXT:   dpct::get_default_queue().submit(
  // CHECK-NEXT:     [&](sycl::handler &cgh) {
  // CHECK-NEXT:       auto const_angle_acc_ct1 = const_angle.get_access(cgh);
  // CHECK-NEXT:       auto const_ptr_acc_ct1 = const_ptr.get_access(cgh);
  // CHECK-NEXT:       auto d_array_acc_ct0 = d_array_buf_ct0.get_access<sycl::access::mode::read_write>(cgh);
  // CHECK-EMPTY:
  // CHECK-NEXT:       cgh.parallel_for<dpct_kernel_name<class simple_kernel_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:         sycl::nd_range<3>(sycl::range<3>(1, 1, size / 64) * sycl::range<3>(1, 1, 64), sycl::range<3>(1, 1, 64)),
  // CHECK-NEXT:         [=](sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:           simple_kernel((float *)(&d_array_acc_ct0[0]), item_ct1, const_angle_acc_ct1.get_pointer(), const_ptr_acc_ct1.get_pointer());
  // CHECK-NEXT:         });
  // CHECK-NEXT:     });
  // CHECK-NEXT: }
  simple_kernel<<<size / 64, 64>>>(d_array);

  float hangle_h[360];
  // CHECK:  dpct::dpct_memcpy(hangle_h, d_array, 360 * sizeof(float), dpct::device_to_host);
  cudaMemcpy(hangle_h, d_array, 360 * sizeof(float), cudaMemcpyDeviceToHost);
  for (int i = 0; i < 360; i++) {
    if (fabs(h_array[i] - hangle_h[i]) > 1e-5) {
      exit(-1);
    }
  }

  h_array[0] = 10.0f; // Just to test
  // CHECK: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT:  (dpct::dpct_memcpy(const_one.get_ptr(), &h_array[0], sizeof(float) * 1), 0);
  cudaMemcpyToSymbol(&const_one, &h_array[0], sizeof(float) * 1);

  // CHECK: {
  // CHECK-NEXT:   dpct::buffer_t d_array_buf_ct0 = dpct::get_buffer(d_array);
  // CHECK-NEXT:   dpct::get_default_queue().submit(
  // CHECK-NEXT:     [&](sycl::handler &cgh) {
  // CHECK-NEXT:       auto const_float_acc_ct1 = const_float.get_access(cgh);
  // CHECK-NEXT:       auto const_one_acc_ct1 = const_one.get_access(cgh);
  // CHECK-NEXT:       auto d_array_acc_ct0 = d_array_buf_ct0.get_access<sycl::access::mode::read_write>(cgh);
  // CHECK-EMPTY:
  // CHECK-NEXT:       cgh.parallel_for<dpct_kernel_name<class simple_kernel_one_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:         sycl::nd_range<3>(sycl::range<3>(1, 1, size / 64) * sycl::range<3>(1, 1, 64), sycl::range<3>(1, 1, 64)),
  // CHECK-NEXT:         [=](sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:           simple_kernel_one((float *)(&d_array_acc_ct0[0]), item_ct1, dpct::accessor<float, dpct::constant, 2>(const_float_acc_ct1), const_one_acc_ct1);
  // CHECK-NEXT:         });
  // CHECK-NEXT:     });
  // CHECK-NEXT: }
  simple_kernel_one<<<size / 64, 64>>>(d_array);

  // CHECK:  dpct::dpct_memcpy(hangle_h, d_array, 360 * sizeof(float), dpct::device_to_host);
  cudaMemcpy(hangle_h, d_array, 360 * sizeof(float), cudaMemcpyDeviceToHost);

  for (int i = 1; i < 360; i++) {
    if (fabs(h_array[i] + 30.0f - hangle_h[i]) > 1e-5) {
      exit(-1);
    }
  }

  cudaFree(d_array);

  printf("Test Passed!\n");
  return 0;
}


// CHECK: dpct::constant_memory<float, 0> C;
__constant__ float C;

// CHECK: void foo(float d, float y, float C){
// CHECK-NEXT:   float temp;
// CHECK-NEXT:   float maxtemp = sycl::fmax(temp=(y*d)<(y==1?C:0) ? -(3*y) :-10, (float)(-10));
// CHECK-NEXT: }
__global__ void foo(float d, float y){
  float temp;
  float maxtemp = fmaxf(temp=(y*d)<(y==1?C:0) ? -(3*y) :-10, -10);
}
