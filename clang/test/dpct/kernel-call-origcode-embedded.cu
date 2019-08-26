// RUN: dpct -keep-original-code -out-root %T %s -- -x cuda --cuda-host-only --cuda-path="%cuda-path"
// RUN: FileCheck --input-file %T/kernel-call-origcode-embedded.dp.cpp --match-full-lines %s

#include <iostream>
// includes CUDA
// CHECK:  /* DPCT_ORIG #include <cuda_runtime.h>*/
#include <cuda_runtime.h>

// CHECK:   /* DPCT_ORIG __global__ void testKernelPtr(const int *L, const int *M, int N) {*/
// CHECK-NEXT:void testKernelPtr(const int *L, const int *M, int N, cl::sycl::nd_item<3> [[ITEMNAME:item_ct1]]) {
__global__ void testKernelPtr(const int *L, const int *M, int N) {

  // CHECK: /* DPCT_ORIG   int gtid = blockIdx.x  * blockDim.x */
  // CHECK-NEXT:   int gtid = item_ct1.get_group(0) /*comments*/ * item_ct1.get_local_range().get(0) /*comments
  // CHECK-NEXT:  comments*/
  // CHECK-NEXT: /* DPCT_ORIG   + threadIdx.x;*/
  // CHECK-NEXT:  + item_ct1.get_local_id(0);
  int gtid = blockIdx.x /*comments*/ * blockDim.x /*comments
  comments*/
             + threadIdx.x;
}

// CHECK:     /* DPCT_ORIG __global__ void testKernel(int L, int M, int N) {*/
// CHECK-NEXT: void testKernel(int L, int M, int N, cl::sycl::nd_item<3> [[ITEMNAME:item_ct1]]) {
__global__ void testKernel(int L, int M, int N) {
  // CHECK:      /* DPCT_ORIG   int gtid = blockIdx.x*/
  // CHECK-NEXT:  int gtid = item_ct1.get_group(0)
  // CHECK-NEXT: /* DPCT_ORIG              * blockDim.x*/
  // CHECK-NEXT:                * item_ct1.get_local_range().get(0)
  // CHECK-NEXT: /* DPCT_ORIG              + threadIdx.x;*/
  // CHECK-NEXT:                + item_ct1.get_local_id(0);
  int gtid = blockIdx.x
             * blockDim.x
             + threadIdx.x;
}

// Error handling macro

// CHECK: #define CUDA_CHECK(call) \
// CHECK-NEXT:  /* DPCT_ORIG     if ((call) != cudaSuccess) { \*/ \
// CHECK-NEXT:      if ((call) != 0) { \
// CHECK-NEXT:  /* DPCT_ORIG         cudaError_t err = cudaGetLastError(); \*/ \
// CHECK-NEXT:          int err = 0; \
// CHECK-NEXT:          std::cout << "CUDA error calling \"" #call "\", code is " << err << std::endl; \
// CHECK-NEXT:          exit(err); \
// CHECK-NEXT:       }
#define CUDA_CHECK(call)                                                           \
    if ((call) != cudaSuccess) { \
        cudaError_t err = cudaGetLastError(); \
        std::cout << "CUDA error calling \"" #call "\", code is " << err << std::endl; \
        exit(err); \
    }

#define checkCudaErrors(val) check((val), #val, __FILE__, __LINE__)
template <typename T>
void check(T result, char const *const func, const char *const file, int const line) {}

int main() {
  // CHECK:  /* DPCT_ORIG   dim3 griddim = 2;*/
  // CHECK-NEXT:  cl::sycl::range<3> griddim = cl::sycl::range<3>(2, 1, 1);
  dim3 griddim = 2;

  // CHECK:  /* DPCT_ORIG   dim3 threaddim = 32;*/
  // CHECK-NEXT:   cl::sycl::range<3> threaddim = cl::sycl::range<3>(32, 1, 1);
  dim3 threaddim = 32;

  void *karg1 = 0;
  const int *karg2 = 0;
  int karg3 = 80;
  // CHECK:  /* DPCT_ORIG   testKernelPtr<<<griddim, threaddim>>>((const int *)karg1,
  // CHECK-NEXT:  karg2, karg3);*/
  // CHECK-NEXT: {
  // CHECK-NEXT:   std::pair<dpct::buffer_t, size_t> arg_ct0_buf = dpct::get_buffer_and_offset((const int *)karg1);
  // CHECK-NEXT:   size_t arg_ct0_offset = arg_ct0_buf.second;
  // CHECK-NEXT:   std::pair<dpct::buffer_t, size_t> arg_ct1_buf = dpct::get_buffer_and_offset(karg2);
  // CHECK-NEXT:   size_t arg_ct1_offset = arg_ct1_buf.second;
  // CHECK-NEXT:   dpct::get_default_queue().submit(
  // CHECK-NEXT:     [&](cl::sycl::handler &cgh) {
  // CHECK-NEXT:       auto arg_ct0_acc = arg_ct0_buf.first.get_access<cl::sycl::access::mode::read_write>(cgh);
  // CHECK-NEXT:       auto arg_ct1_acc = arg_ct1_buf.first.get_access<cl::sycl::access::mode::read_write>(cgh);
  // CHECK-NEXT:       cgh.parallel_for<dpct_kernel_name<class testKernelPtr_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:         cl::sycl::nd_range<3>((griddim * threaddim), threaddim),
  // CHECK-NEXT:         [=](cl::sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:           const int *arg_ct0 = (const int *)(&arg_ct0_acc[0] + arg_ct0_offset);
  // CHECK-NEXT:           const int *arg_ct1 = (const int *)(&arg_ct1_acc[0] + arg_ct1_offset);
  // CHECK-NEXT:           testKernelPtr(arg_ct0, arg_ct1, karg3, item_ct1);
  // CHECK-NEXT:         });
  // CHECK-NEXT:     });
  // CHECK-NEXT: }
  testKernelPtr<<<griddim, threaddim>>>((const int *)karg1,
                                        karg2, karg3);


  int karg1int = 1;
  int karg2int = 2;
  int karg3int = 3;
  int intvar = 20;
  // CHECK: /* DPCT_ORIG   testKernel<<<10, intvar>>>(karg1int, karg2int,
  // CHECK:  karg3int);*/
  // CHECK-NEXT: {
  // CHECK-NEXT:   dpct::get_default_queue().submit(
  // CHECK-NEXT:     [&](cl::sycl::handler &cgh) {
  // CHECK-NEXT:       cgh.parallel_for<dpct_kernel_name<class testKernel_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:         cl::sycl::nd_range<3>((cl::sycl::range<3>(10, 1, 1) * cl::sycl::range<3>(intvar, 1, 1)), cl::sycl::range<3>(intvar, 1, 1)),
  // CHECK-NEXT:         [=](cl::sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:           testKernel(karg1int, karg2int, karg3int, item_ct1);
  // CHECK-NEXT:         });
  // CHECK-NEXT:     });
  // CHECK-NEXT: }
  testKernel<<<10, intvar>>>(karg1int, karg2int, // comments
                             // comments.
                             karg3int);

  // CHECK: /* DPCT_ORIG   testKernel<<<dim3(1), dim3(1, 2)>>>(karg1int,
  // CHECK:  karg2int,
  // CHECK:  karg3int);*/
  // CHECK-NEXT: {
  // CHECK-NEXT:   dpct::get_default_queue().submit(
  // CHECK-NEXT:     [&](cl::sycl::handler &cgh) {
  // CHECK-NEXT:       cgh.parallel_for<dpct_kernel_name<class testKernel_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:         cl::sycl::nd_range<3>((cl::sycl::range<3>(1, 1, 1) * cl::sycl::range<3>(1, 2, 1)), cl::sycl::range<3>(1, 2, 1)),
  // CHECK-NEXT:         [=](cl::sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:           testKernel(karg1int, karg2int, karg3int, item_ct1);
  // CHECK-NEXT:         });
  // CHECK-NEXT:     });
  // CHECK-NEXT: }
  testKernel<<<dim3(1), dim3(1, 2)>>>(karg1int,
                                      /* comments */
                                      karg2int, // comments
                                      /*
                                      comments
                                      */
                                      karg3int);

  // CHECK: /* DPCT_ORIG   testKernel<<<dim3(1, 2), dim3(1, 2, 3)>>>(karg1int,
  // CHECK-NEXT:  karg2int, karg3int); */
  // CHECK-NEXT: {
  // CHECK-NEXT:   dpct::get_default_queue().submit(
  // CHECK-NEXT:     [&](cl::sycl::handler &cgh) {
  // CHECK-NEXT:       cgh.parallel_for<dpct_kernel_name<class testKernel_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:         cl::sycl::nd_range<3>((cl::sycl::range<3>(1, 2, 1) * cl::sycl::range<3>(1, 2, 3)), cl::sycl::range<3>(1, 2, 3)),
  // CHECK-NEXT:         [=](cl::sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:           testKernel(karg1int, karg2int, karg3int, item_ct1);
  // CHECK-NEXT:         });
  // CHECK-NEXT:     });
  // CHECK-NEXT: }
  testKernel<<<dim3(1, 2), dim3(1, 2, 3)>>>(karg1int,
	  karg2int, /* comments */karg3int/* comments */); // comments

  // CHECK: /* DPCT_ORIG   testKernel<<<griddim.x, griddim.y + 2>>>(karg1int, karg2int, karg3int);*/
  // CHECK-NEXT: {
  // CHECK-NEXT:   dpct::get_default_queue().submit(
  // CHECK-NEXT:     [&](cl::sycl::handler &cgh) {
  // CHECK-NEXT:       cgh.parallel_for<dpct_kernel_name<class testKernel_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:         cl::sycl::nd_range<3>((cl::sycl::range<3>(griddim[0], 1, 1) * cl::sycl::range<3>(griddim[1] + 2, 1, 1)), cl::sycl::range<3>(griddim[1] + 2, 1, 1)),
  // CHECK-NEXT:         [=](cl::sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:           testKernel(karg1int, karg2int, karg3int, item_ct1);
  // CHECK-NEXT:         });
  // CHECK-NEXT:     });
  // CHECK-NEXT: }
  testKernel<<<griddim.x, griddim.y + 2>>>(karg1int, karg2int, karg3int);

  float *deviceOutputData = NULL;

  // CHECK: /* DPCT_ORIG   CUDA_CHECK(cudaMalloc((void **)&deviceOutputData, 10 * sizeof(float)));*/
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  CUDA_CHECK(cudaMalloc((void **)&deviceOutputData, 10 * sizeof(float)));

  // copy result from device to host
  float *h_odata = NULL;
  float *d_odata = NULL;
  // CHECK: /* DPCT_ORIG   checkCudaErrors(cudaMemcpy(h_odata, d_odata, sizeof(float) * 4, cudaMemcpyDeviceToHost));*/
  // CHECK-NEXT:/*
  // CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT:*/
  checkCudaErrors(cudaMemcpy(h_odata, d_odata, sizeof(float) * 4, cudaMemcpyDeviceToHost));

  // CHECK: /*
  // CHECK-NEXT:DPCT1007:{{[0-9]+}}: Migration of this CUDA API is not supported by the oneAPI DPC++ Compatibility Tool.
  // CHECK-NEXT:*/
  cudaThreadGetCacheConfig(NULL);

  // CHECK: /* DPCT_ORIG   cudaThreadGetCacheConfig(NULL);cudaMalloc((void **)&deviceOutputData, 10 * sizeof(float));*/
  // CHECK-NEXT: /*
  // CHECK-NEXT:  DPCT1007:{{[0-9]+}}: Migration of this CUDA API is not supported by the oneAPI DPC++ Compatibility Tool.
  // CHECK-NEXT: */
  cudaThreadGetCacheConfig(NULL);cudaMalloc((void **)&deviceOutputData, 10 * sizeof(float));

  // CHECK: /* DPCT_ORIG   cudaEventCreate(NULL);checkCudaErrors(cudaMemcpy(h_odata, d_odata, sizeof(float) * 4, cudaMemcpyDeviceToHost));checkCudaErrors(cudaMemcpy(h_odata, d_odata, sizeof(float) * 4, cudaMemcpyDeviceToHost));*/
  // CHECK-NEXT:  /*
  // CHECK-NEXT:  DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT:  */
  // CHECK-NEXT:  /*
  // CHECK-NEXT:  DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT:  */
  cudaEventCreate(NULL);checkCudaErrors(cudaMemcpy(h_odata, d_odata, sizeof(float) * 4, cudaMemcpyDeviceToHost));checkCudaErrors(cudaMemcpy(h_odata, d_odata, sizeof(float) * 4, cudaMemcpyDeviceToHost));
}
