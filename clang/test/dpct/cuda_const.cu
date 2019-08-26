// RUN: dpct --usm-level=none -out-root %T %s -- -x cuda --cuda-host-only --cuda-path="%cuda-path"
// RUN: FileCheck %s --match-full-lines --input-file %T/cuda_const.dp.cpp

#include <stdio.h>

#define NUM_ELEMENTS 16
const unsigned num_elements = 16;
// CHECK: dpct::constant_memory<float, 1> const_angle(360);
// CHECK: dpct::constant_memory<float, 2> const_float(NUM_ELEMENTS, num_elements * 2);
__constant__ float const_angle[360], const_float[NUM_ELEMENTS][num_elements * 2];
// CHECK: dpct::constant_memory<cl::sycl::double2, 0> vec_d;
__constant__ double2 vec_d;

// CHECK:void simple_kernel(float *d_array, cl::sycl::nd_item<3> [[ITEM:item_ct1]], dpct::dpct_accessor<float, dpct::constant, 1> const_angle) {
// CHECK-NEXT:  int index;
// CHECK-NEXT:  index = [[ITEM]].get_group(0) * [[ITEM]].get_local_range().get(0) + [[ITEM]].get_local_id(0);
// CHECK-NEXT:  if (index < 360) {
// CHECK-NEXT:    d_array[index] = const_angle[index];
// CHECK-NEXT:  }
// CHECK-NEXT:  return;
// CHECK-NEXT:}
__global__ void simple_kernel(float *d_array) {
  int index;
  index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < 360) {
    d_array[index] = const_angle[index];
  }
  return;
}

// CHECK: dpct::constant_memory<float, 0> const_one;
__constant__ float const_one;

// CHECK:void simple_kernel_one(float *d_array, cl::sycl::nd_item<3> [[ITEM:item_ct1]], dpct::dpct_accessor<float, dpct::constant, 2> const_float, dpct::dpct_accessor<float, dpct::constant, 0> const_one) {
// CHECK-NEXT:  int index;
// CHECK-NEXT:  index = [[ITEM]].get_group(0) * [[ITEM]].get_local_range().get(0) + [[ITEM]].get_local_id(0);
// CHECK-NEXT:  if (index < 33) {
// CHECK-NEXT:    d_array[index] = (float)const_one + const_float[index][index];
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
  float *d_array;
  float h_array[360];

  // CHECK: dpct::dpct_malloc((void **)&d_array, sizeof(float) * size);
  cudaMalloc((void **)&d_array, sizeof(float) * size);

  // CHECK: dpct::dpct_memset((void*)(d_array), 0, sizeof(float) * size);
  cudaMemset(d_array, 0, sizeof(float) * size);

  for (int loop = 0; loop < 360; loop++)
    h_array[loop] = acos(-1.0f) * loop / 180.0f;

  // CHECK:/*
  // CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT:*/
  // CHECK-NEXT:   (dpct::dpct_memcpy(const_angle.get_ptr(), (void*)(&h_array[0]), sizeof(float) * 360), 0);
  cudaMemcpyToSymbol(&const_angle[0], &h_array[0], sizeof(float) * 360);

  // CHECK:/*
  // CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT:*/
  // CHECK-NEXT:   (dpct::dpct_memcpy(const_angle.get_ptr() + sizeof(float) * (3), (void*)(&h_array[0]), sizeof(float) * 357), 0);
  cudaMemcpyToSymbol(&const_angle[3], &h_array[0], sizeof(float) * 357);

  // CHECK:  /*
  // CHECK-NEXT:  DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT:  */
  // CHECK-NEXT:  (dpct::dpct_memcpy((void*)(&h_array[0]), const_angle.get_ptr() + sizeof(float) * (3), sizeof(float) * 357), 0);
  cudaMemcpyFromSymbol(&h_array[0], &const_angle[3], sizeof(float) * 357);

  #define NUM 3
  // CHECK:/*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: (dpct::dpct_memcpy(const_angle.get_ptr() + sizeof(float) * (3+NUM), (void*)(&h_array[0]), sizeof(float) * 354), 0);
  cudaMemcpyToSymbol(&const_angle[3+NUM], &h_array[0], sizeof(float) * 354);

  // CHECK:  /*
  // CHECK-NEXT:  DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT:  */
  // CHECK-NEXT:  (dpct::dpct_memcpy((void*)(&h_array[0]), const_angle.get_ptr() + sizeof(float) * (3+NUM), sizeof(float) * 354), 0);
  cudaMemcpyFromSymbol(&h_array[0], &const_angle[3+NUM], sizeof(float) * 354);
  // CHECK: {
  // CHECK-NEXT:   std::pair<dpct::buffer_t, size_t> arg_ct0_buf = dpct::get_buffer_and_offset(d_array);
  // CHECK-NEXT:   size_t arg_ct0_offset = arg_ct0_buf.second;
  // CHECK-NEXT:   dpct::get_default_queue().submit(
  // CHECK-NEXT:     [&](cl::sycl::handler &cgh) {
  // CHECK-NEXT:       auto const_angle_acc_ct1 = const_angle.get_access(cgh);
  // CHECK-NEXT:       auto arg_ct0_acc = arg_ct0_buf.first.get_access<cl::sycl::access::mode::read_write>(cgh);
  // CHECK-NEXT:       cgh.parallel_for<dpct_kernel_name<class simple_kernel_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:         cl::sycl::nd_range<3>((cl::sycl::range<3>(size / 64, 1, 1) * cl::sycl::range<3>(64, 1, 1)), cl::sycl::range<3>(64, 1, 1)),
  // CHECK-NEXT:         [=](cl::sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:           float *arg_ct0 = (float *)(&arg_ct0_acc[0] + arg_ct0_offset);
  // CHECK-NEXT:           simple_kernel(arg_ct0, item_ct1, dpct::dpct_accessor<float, dpct::constant, 1>(const_angle_acc_ct1));
  // CHECK-NEXT:         });
  // CHECK-NEXT:     });
  // CHECK-NEXT: }
  simple_kernel<<<size / 64, 64>>>(d_array);

  float hangle_h[360];
  // CHECK:  dpct::dpct_memcpy((void*)(hangle_h), (void*)(d_array), 360 * sizeof(float), dpct::device_to_host);
  cudaMemcpy(hangle_h, d_array, 360 * sizeof(float), cudaMemcpyDeviceToHost);
  for (int i = 0; i < 360; i++) {
    if (fabs(h_array[i] - hangle_h[i]) > 1e-5) {
      exit(-1);
    }
  }

  h_array[0] = 10.0f; // Just to test
  // CHECK: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT:  (dpct::dpct_memcpy(const_one.get_ptr(), (void*)(&h_array[0]), sizeof(float) * 1), 0);
  cudaMemcpyToSymbol(&const_one, &h_array[0], sizeof(float) * 1);

  // CHECK: {
  // CHECK-NEXT:   std::pair<dpct::buffer_t, size_t> arg_ct0_buf = dpct::get_buffer_and_offset(d_array);
  // CHECK-NEXT:   size_t arg_ct0_offset = arg_ct0_buf.second;
  // CHECK-NEXT:   dpct::get_default_queue().submit(
  // CHECK-NEXT:     [&](cl::sycl::handler &cgh) {
  // CHECK-NEXT:       auto const_float_acc_ct1 = const_float.get_access(cgh);
  // CHECK-NEXT:       auto const_one_acc_ct1 = const_one.get_access(cgh);
  // CHECK-NEXT:       auto arg_ct0_acc = arg_ct0_buf.first.get_access<cl::sycl::access::mode::read_write>(cgh);
  // CHECK-NEXT:       cgh.parallel_for<dpct_kernel_name<class simple_kernel_one_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:         cl::sycl::nd_range<3>((cl::sycl::range<3>(size / 64, 1, 1) * cl::sycl::range<3>(64, 1, 1)), cl::sycl::range<3>(64, 1, 1)),
  // CHECK-NEXT:         [=](cl::sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:           float *arg_ct0 = (float *)(&arg_ct0_acc[0] + arg_ct0_offset);
  // CHECK-NEXT:           simple_kernel_one(arg_ct0, item_ct1, dpct::dpct_accessor<float, dpct::constant, 2>(const_float_acc_ct1), dpct::dpct_accessor<float, dpct::constant, 0>(const_one_acc_ct1));
  // CHECK-NEXT:         });
  // CHECK-NEXT:     });
  // CHECK-NEXT: }
  simple_kernel_one<<<size / 64, 64>>>(d_array);

  hangle_h[360];
  // CHECK:  dpct::dpct_memcpy((void*)(hangle_h), (void*)(d_array), 360 * sizeof(float), dpct::device_to_host);
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
