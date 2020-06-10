// RUN: dpct --format-range=none --usm-level=none -out-root %T %s --cuda-include-path="%cuda-path/include"  -- -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/cublas-create-Sgemm-destroy.dp.cpp --match-full-lines %s
// CHECK: #include <CL/sycl.hpp>
// CHECK-NEXT: #include <dpct/dpct.hpp>
// CHECK-NEXT: #include <cstdio>
// CHECK: #include <mkl_blas_sycl.hpp>
// CHECK-NEXT: #include <mkl_lapack_sycl.hpp>
// CHECK-NEXT: #include <mkl_sycl_types.hpp>
#include <cstdio>
#include "cublas_v2.h"
#include <cuda_runtime.h>

void foo (cublasStatus_t s){
}
cublasStatus_t bar (cublasStatus_t s){
  return s;
}

// CHECK: extern sycl::queue* handle2;
extern cublasHandle_t handle2;

int main() {
  // CHECK: dpct::device_ext &dev_ct1 = dpct::get_current_device();
  // CHECK-NEXT: sycl::queue &q_ct1 = dev_ct1.default_queue();
  // CHECK: int status;
  // CHECK-NEXT: sycl::queue* handle;
  // CHECK-NEXT: handle = &q_ct1;
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: status = (handle = &q_ct1, 0);
  // CHECK-NEXT: if (status != 0) {
  cublasStatus_t status;
  cublasHandle_t handle;
  cublasCreate(&handle);
  status = cublasCreate(&handle);
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf(stderr, "!!!! CUBLAS initialization error\n");
    return EXIT_FAILURE;
  }

  //CHECK: int mode = 0;
  //CHECK-NEXT: /*
  //CHECK-NEXT: DPCT1026:{{[0-9]+}}: The call to cublasGetPointerMode was removed, because the function call is redundant in DPC++.
  //CHECK-NEXT: */
  //CHECK-NEXT: /*
  //CHECK-NEXT: DPCT1026:{{[0-9]+}}: The call to cublasSetPointerMode was removed, because the function call is redundant in DPC++.
  //CHECK-NEXT: */
  cublasPointerMode_t mode = CUBLAS_POINTER_MODE_HOST;
  cublasGetPointerMode(handle, &mode);
  cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE);

  // CHECK: sycl::queue *stream1;
  // CHECK-NEXT: stream1 = dev_ct1.create_queue();
  // CHECK-NEXT: handle = stream1;
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: status = (handle = stream1, 0);
  // CHECK-NEXT: stream1 = handle;
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: status = (stream1 = handle, 0);
  cudaStream_t stream1;
  cudaStreamCreate(&stream1);
  cublasSetStream(handle, stream1);
  status = cublasSetStream(handle, stream1);
  cublasGetStream(handle, &stream1);
  status = cublasGetStream(handle, &stream1);


  int N = 275;
  float *d_A_S = 0;
  float *d_B_S = 0;
  float *d_C_S = 0;
  float alpha_S = 1.0f;
  float beta_S = 0.0f;
  int trans0 = 0;
  int trans1 = 1;
  int trans2 = 2;
  // CHECK: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK: status = (mkl::blas::gemm(*handle, mkl::transpose::nontrans, mkl::transpose::nontrans, N, N, N, dpct::get_value(&alpha_S, *handle), d_A_S_buf_ct{{[0-9]+}}, N, d_B_S_buf_ct{{[0-9]+}}, N, dpct::get_value(&beta_S, *handle), d_C_S_buf_ct{{[0-9]+}}, N), 0);
  // CHECK: mkl::blas::gemm(*handle, trans0==2 ? mkl::transpose::conjtrans : (mkl::transpose)trans0, trans1==2 ? mkl::transpose::conjtrans : (mkl::transpose)trans1, N, N, N, dpct::get_value(&alpha_S, *handle), d_A_S_buf_ct{{[0-9]+}}, N, d_B_S_buf_ct{{[0-9]+}}, N, dpct::get_value(&beta_S, *handle), d_C_S_buf_ct{{[0-9]+}}, N);
  status = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha_S, d_A_S, N, d_B_S, N, &beta_S, d_C_S, N);
  cublasSgemm(handle, (cublasOperation_t)trans0, (cublasOperation_t)trans1, N, N, N, &alpha_S, d_A_S, N, d_B_S, N, &beta_S, d_C_S, N);
  double *d_A_D = 0;
  double *d_B_D = 0;
  double *d_C_D = 0;
  double alpha_D = 1.0;
  double beta_D = 0.0;
  // CHECK: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK: status = (mkl::blas::gemm(*handle, mkl::transpose::nontrans, mkl::transpose::nontrans, N, N, N, dpct::get_value(&alpha_D, *handle), d_A_D_buf_ct{{[0-9]+}}, N, d_B_D_buf_ct{{[0-9]+}}, N, dpct::get_value(&beta_D, *handle), d_C_D_buf_ct{{[0-9]+}}, N), 0);
  // CHECK: mkl::blas::gemm(*handle, trans2==2 ? mkl::transpose::conjtrans : (mkl::transpose)trans2, mkl::transpose::conjtrans, N, N, N, dpct::get_value(&alpha_D, *handle), d_A_D_buf_ct{{[0-9]+}}, N, d_B_D_buf_ct{{[0-9]+}}, N, dpct::get_value(&beta_D, *handle), d_C_D_buf_ct{{[0-9]+}}, N);
  status = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha_D, d_A_D, N, d_B_D, N, &beta_D, d_C_D, N);
  cublasDgemm(handle, (cublasOperation_t)trans2, (cublasOperation_t)2, N, N, N, &alpha_D, d_A_D, N, d_B_D, N, &beta_D, d_C_D, N);

  __half *d_A_H = 0;
  __half *d_B_H = 0;
  __half *d_C_H = 0;
  __half alpha_H;
  __half beta_H;
  // CHECK: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK: status = (mkl::blas::gemm(*handle, mkl::transpose::nontrans, mkl::transpose::nontrans, N, N, N, dpct::get_value(&alpha_H, *handle), d_A_H_buf_ct{{[0-9]+}}, N, d_B_H_buf_ct{{[0-9]+}}, N, dpct::get_value(&beta_H, *handle), d_C_H_buf_ct{{[0-9]+}}, N), 0);
  // CHECK: mkl::blas::gemm(*handle, trans2==2 ? mkl::transpose::conjtrans : (mkl::transpose)trans2, mkl::transpose::conjtrans, N, N, N, dpct::get_value(&alpha_H, *handle), d_A_H_buf_ct{{[0-9]+}}, N, d_B_H_buf_ct{{[0-9]+}}, N, dpct::get_value(&beta_H, *handle), d_C_H_buf_ct{{[0-9]+}}, N);
  status = cublasHgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha_H, d_A_H, N, d_B_H, N, &beta_H, d_C_H, N);
  cublasHgemm(handle, (cublasOperation_t)trans2, (cublasOperation_t)2, N, N, N, &alpha_H, d_A_H, N, d_B_H, N, &beta_H, d_C_H, N);

  // CHECK: for (;;) {
  // CHECK-NEXT: {
  // CHECK-NEXT: auto d_A_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(d_A_S);
  // CHECK-NEXT: auto d_B_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(d_B_S);
  // CHECK-NEXT: auto d_C_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(d_C_S);
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: status = (mkl::blas::gemm(*handle, mkl::transpose::trans, mkl::transpose::trans, N, N, N, dpct::get_value(&alpha_S, *handle), d_A_S_buf_ct{{[0-9]+}}, N, d_B_S_buf_ct{{[0-9]+}}, N, dpct::get_value(&beta_S, *handle), d_C_S_buf_ct{{[0-9]+}}, N), 0);
  // CHECK-NEXT: }
  // CHECK-NEXT: beta_S = beta_S + 1;
  // CHECK-NEXT: }
  // CHECK-NEXT: alpha_S = alpha_S + 1;
  for (;;) {
    status = cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, N, N, N, &alpha_S, d_A_S, N, d_B_S, N, &beta_S, d_C_S, N);
    beta_S = beta_S + 1;
  }
  alpha_S = alpha_S + 1;

  // CHECK: for (;;) {
  // CHECK-NEXT: {
  // CHECK-NEXT: auto d_A_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(d_A_S);
  // CHECK-NEXT: auto d_B_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(d_B_S);
  // CHECK-NEXT: auto d_C_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(d_C_S);
  // CHECK-NEXT: mkl::blas::gemm(*handle, mkl::transpose::trans, mkl::transpose::trans, N, N, N, dpct::get_value(&alpha_S, *handle), d_A_S_buf_ct{{[0-9]+}}, N, d_B_S_buf_ct{{[0-9]+}}, N, dpct::get_value(&beta_S, *handle), d_C_S_buf_ct{{[0-9]+}}, N);
  // CHECK-NEXT: }
  // CHECK-NEXT: beta_S = beta_S + 1;
  // CHECK-NEXT: }
  // CHECK-NEXT: alpha_S = alpha_S + 1;
  for (;;) {
    cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, N, N, N, &alpha_S, d_A_S, N, d_B_S, N, &beta_S, d_C_S, N);
    beta_S = beta_S + 1;
  }
  alpha_S = alpha_S + 1;


  // CHECK: {
  // CHECK-NEXT: auto d_A_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(d_A_S);
  // CHECK-NEXT: auto d_B_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(d_B_S);
  // CHECK-NEXT: auto d_C_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(d_C_S);
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: foo(bar((mkl::blas::gemm(*handle, mkl::transpose::nontrans, mkl::transpose::nontrans, N, N, N, dpct::get_value(&alpha_S, *handle), d_A_S_buf_ct{{[0-9]+}}, N, d_B_S_buf_ct{{[0-9]+}}, N, dpct::get_value(&beta_S, *handle), d_C_S_buf_ct{{[0-9]+}}, N), 0)));
  foo(bar(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha_S, d_A_S, N, d_B_S, N, &beta_S, d_C_S, N)));

  // CHECK: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: status = (handle = nullptr, 0);
  // CHECK-NEXT: handle = nullptr;
  // CHECK-NEXT: return 0;
  status = cublasDestroy(handle);
  cublasDestroy(handle);
  return 0;
}
