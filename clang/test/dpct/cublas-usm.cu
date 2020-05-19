// RUN: dpct --format-range=none -out-root %T %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/cublas-usm.dp.cpp --match-full-lines %s
#include <cstdio>
#include <cublas_v2.h>
#include <cuda_runtime.h>

cublasHandle_t handle;
int N = 275;
float *h_a, *h_b, *h_c;
const float *d_A_S;
const float *d_B_S;
float *d_C_S;
float alpha_S = 1.0f;
float beta_S = 0.0f;
int trans0 = 0;
int trans1 = 1;
int trans2 = 2;
int fill0 = 0;
int side0 = 0;
int diag0 = 0;
int *result = 0;
const float *x_S = 0;
const float *y_S = 0;

const double *d_A_D;
const double  *d_B_D;
double  *d_C_D;
double alpha_D;
double beta_D;
const double *x_D;
const double *y_D;

const float2 *d_A_C;
const float2  *d_B_C;
float2  *d_C_C;
float2 alpha_C;
float2 beta_C;
const float2 *x_C;
const float2 *y_C;

const double2 *d_A_Z;
const double2  *d_B_Z;
double2  *d_C_Z;
double2 alpha_Z;
double2 beta_Z;
const double2 *x_Z;
const double2 *y_Z;

float* result_S;
double* result_D;
float2* result_C;
double2* result_Z;

int incx, incy, lda, ldb, ldc;

int main() {

  //CHECK:/*
  //CHECK-NEXT:DPCT1018:{{[0-9]+}}: The cublasSetVector was migrated, but due to parameter 11111 equals to parameter 11111 but greater than 1, the generated code performance may be sub-optimal.
  //CHECK-NEXT:*/
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:int a = (dpct::matrix_mem_copy((void*)d_C_S, (void*)h_a, 11111, 11111, 1, 10, sizeof(float)), 0);
  //CHECK-NEXT:dpct::matrix_mem_copy((void*)d_C_S, (void*)h_b, 1, 1, 1, 10, sizeof(float));
  //CHECK-NEXT:dpct::matrix_mem_copy((void*)d_C_S, (void*)h_c, 1, 1, 1, 10, sizeof(float));
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:a = (dpct::matrix_mem_copy((void*)d_C_S, (void*)h_a, 100, 100, 100, 100, 10000), 0);
  int a = cublasSetVector(10, sizeof(float), h_a, 11111, d_C_S, 11111);
  cublasSetVector(10, sizeof(float), h_b, 1, d_C_S, 1);
  cublasSetVector(10, sizeof(float), h_c, 1, d_C_S, 1);
  a = cublasSetMatrix(100, 100, 10000, h_a, 100, d_C_S, 100);


  //level 1

  //CHECK:int64_t* res_temp_ptr_ct{{[0-9]+}} = sycl::malloc_shared<int64_t>(1, dpct::get_default_queue());
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:a = (mkl::blas::iamax(*handle, N, x_S, N, res_temp_ptr_ct{{[0-9]+}}).wait(), 0);
  //CHECK-NEXT:*result = (int)*res_temp_ptr_ct{{[0-9]+}};
  //CHECK-NEXT:sycl::free(res_temp_ptr_ct{{[0-9]+}}, dpct::get_default_queue());
  a = cublasIsamax(handle, N, x_S, N, result);
  //CHECK:int64_t* res_temp_ptr_ct{{[0-9]+}} = sycl::malloc_shared<int64_t>(1, dpct::get_default_queue());
  //CHECK-NEXT:mkl::blas::iamax(*handle, N, x_D, N, res_temp_ptr_ct{{[0-9]+}}).wait();
  //CHECK-NEXT:*result = (int)*res_temp_ptr_ct{{[0-9]+}};
  //CHECK-NEXT:sycl::free(res_temp_ptr_ct{{[0-9]+}}, dpct::get_default_queue());
  cublasIdamax(handle, N, x_D, N, result);
  //CHECK:int64_t* res_temp_ptr_ct{{[0-9]+}} = sycl::malloc_shared<int64_t>(1, dpct::get_default_queue());
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:a = (mkl::blas::iamax(*handle, N, (std::complex<float>*)x_C, N, res_temp_ptr_ct{{[0-9]+}}).wait(), 0);
  //CHECK-NEXT:*result = (int)*res_temp_ptr_ct{{[0-9]+}};
  //CHECK-NEXT:sycl::free(res_temp_ptr_ct{{[0-9]+}}, dpct::get_default_queue());
  a = cublasIcamax(handle, N, x_C, N, result);
  //CHECK:int64_t* res_temp_ptr_ct{{[0-9]+}} = sycl::malloc_shared<int64_t>(1, dpct::get_default_queue());
  //CHECK-NEXT:mkl::blas::iamax(*handle, N, (std::complex<double>*)x_Z, N, res_temp_ptr_ct{{[0-9]+}}).wait();
  //CHECK-NEXT:*result = (int)*res_temp_ptr_ct{{[0-9]+}};
  //CHECK-NEXT:sycl::free(res_temp_ptr_ct{{[0-9]+}}, dpct::get_default_queue());
  cublasIzamax(handle, N, x_Z, N, result);

  //CHECK:a = (mkl::blas::rotm(*handle, N, d_C_S, N, d_C_S, N, const_cast<float*>(x_S)).wait(), 0);
  a = cublasSrotm(handle, N, d_C_S, N, d_C_S, N, x_S);
  //CHECK:mkl::blas::rotm(*handle, N, d_C_D, N, d_C_D, N, const_cast<double*>(x_D)).wait();
  cublasDrotm(handle, N, d_C_D, N, d_C_D, N, x_D);

  // CHECK: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: a = (mkl::blas::copy(*handle, N, x_S, incx, d_C_S, incy).wait(), 0);
  a = cublasScopy(handle, N, x_S, incx, d_C_S, incy);
  // CHECK:mkl::blas::copy(*handle, N, x_D, incx, d_C_D, incy).wait();
  cublasDcopy(handle, N, x_D, incx, d_C_D, incy);
  // CHECK: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: a = (mkl::blas::copy(*handle, N, (std::complex<float>*)x_C, incx, (std::complex<float>*)d_C_C, incy).wait(), 0);
  a = cublasCcopy(handle, N, x_C, incx, d_C_C, incy);
  // CHECK:mkl::blas::copy(*handle, N, (std::complex<double>*)x_Z, incx, (std::complex<double>*)d_C_Z, incy).wait();
  cublasZcopy(handle, N, x_Z, incx, d_C_Z, incy);


  // CHECK: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: a = (mkl::blas::axpy(*handle, N, alpha_S, x_S, incx, result_S, incy).wait(), 0);
  a = cublasSaxpy(handle, N, &alpha_S, x_S, incx, result_S, incy);
  // CHECK:mkl::blas::axpy(*handle, N, alpha_D, x_D, incx, result_D, incy).wait();
  cublasDaxpy(handle, N, &alpha_D, x_D, incx, result_D, incy);
  // CHECK: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: a = (mkl::blas::axpy(*handle, N, std::complex<float>(alpha_C.x(),alpha_C.y()), (std::complex<float>*)x_C, incx, (std::complex<float>*)result_C, incy).wait(), 0);
  a = cublasCaxpy(handle, N, &alpha_C, x_C, incx, result_C, incy);
  // CHECK:mkl::blas::axpy(*handle, N, std::complex<double>(alpha_Z.x(),alpha_Z.y()), (std::complex<double>*)x_Z, incx, (std::complex<double>*)result_Z, incy).wait();
  cublasZaxpy(handle, N, &alpha_Z, x_Z, incx, result_Z, incy);

  // CHECK: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: a = (mkl::blas::scal(*handle, N, alpha_S, result_S, incx).wait(), 0);
  a = cublasSscal(handle, N, &alpha_S, result_S, incx);
  // CHECK:mkl::blas::scal(*handle, N, alpha_D, result_D, incx).wait();
  cublasDscal(handle, N, &alpha_D, result_D, incx);
  // CHECK: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: a = (mkl::blas::scal(*handle, N, std::complex<float>(alpha_C.x(),alpha_C.y()), (std::complex<float>*)result_C, incx).wait(), 0);
  a = cublasCscal(handle, N, &alpha_C, result_C, incx);
  // CHECK:mkl::blas::scal(*handle, N, std::complex<double>(alpha_Z.x(),alpha_Z.y()), (std::complex<double>*)result_Z, incx).wait();
  cublasZscal(handle, N, &alpha_Z, result_Z, incx);

  // CHECK: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: a = (mkl::blas::nrm2(*handle, N, x_S, incx, result_S).wait(), 0);
  a = cublasSnrm2(handle, N, x_S, incx, result_S);
  // CHECK:mkl::blas::nrm2(*handle, N, x_D, incx, result_D).wait();

  cublasDnrm2(handle, N, x_D, incx, result_D);
  // CHECK: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: a = (mkl::blas::nrm2(*handle, N, (std::complex<float>*)x_C, incx, result_S).wait(), 0);
  a = cublasScnrm2(handle, N, x_C, incx, result_S);
  // CHECK:mkl::blas::nrm2(*handle, N, (std::complex<double>*)x_Z, incx, result_D).wait();
  cublasDznrm2(handle, N, x_Z, incx, result_D);


  //level 2

  // CHECK: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: a = (mkl::blas::gemv(*handle, trans2==2 ? mkl::transpose::conjtrans : (mkl::transpose)trans2, N, N, alpha_S, x_S, lda, y_S, incx, beta_S, result_S, incy).wait(), 0);
  a = cublasSgemv(handle, (cublasOperation_t)trans2, N, N, &alpha_S, x_S, lda, y_S, incx, &beta_S, result_S, incy);
  // CHECK:mkl::blas::gemv(*handle, mkl::transpose::nontrans, N, N, alpha_D, x_D, lda, y_D, incx, beta_D, result_D, incy).wait();
  cublasDgemv(handle, CUBLAS_OP_N, N, N, &alpha_D, x_D, lda, y_D, incx, &beta_D, result_D, incy);
  // CHECK: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: a = (mkl::blas::gemv(*handle, trans2==2 ? mkl::transpose::conjtrans : (mkl::transpose)trans2, N, N, std::complex<float>(alpha_C.x(),alpha_C.y()), (std::complex<float>*)x_C, lda, (std::complex<float>*)y_C, incx, std::complex<float>(beta_C.x(),beta_C.y()), (std::complex<float>*)result_C, incy).wait(), 0);
  a = cublasCgemv(handle, (cublasOperation_t)trans2, N, N, &alpha_C, x_C, lda, y_C, incx, &beta_C, result_C, incy);
  // CHECK:mkl::blas::gemv(*handle, mkl::transpose::nontrans, N, N, std::complex<double>(alpha_Z.x(),alpha_Z.y()), (std::complex<double>*)x_Z, lda, (std::complex<double>*)y_Z, incx, std::complex<double>(beta_Z.x(),beta_Z.y()), (std::complex<double>*)result_Z, incy).wait();
  cublasZgemv(handle, CUBLAS_OP_N, N, N, &alpha_Z, x_Z, lda, y_Z, incx, &beta_Z, result_Z, incy);

  // CHECK: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: a = (mkl::blas::ger(*handle, N, N, alpha_S, x_S, incx, y_S, incy, result_S, lda).wait(), 0);
  a = cublasSger(handle, N, N, &alpha_S, x_S, incx, y_S, incy, result_S, lda);
  // CHECK:mkl::blas::ger(*handle, N, N, alpha_D, x_D, incx, y_D, incy, result_D, lda).wait();
  cublasDger(handle, N, N, &alpha_D, x_D, incx, y_D, incy, result_D, lda);
  // CHECK: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: a = (mkl::blas::geru(*handle, N, N, std::complex<float>(alpha_C.x(),alpha_C.y()), (std::complex<float>*)x_C, incx, (std::complex<float>*)y_C, incy, (std::complex<float>*)result_C, lda).wait(), 0);
  a = cublasCgeru(handle, N, N, &alpha_C, x_C, incx, y_C, incy, result_C, lda);
  // CHECK:mkl::blas::gerc(*handle, N, N, std::complex<float>(alpha_C.x(),alpha_C.y()), (std::complex<float>*)x_C, incx, (std::complex<float>*)y_C, incy, (std::complex<float>*)result_C, lda).wait();
  cublasCgerc(handle, N, N, &alpha_C, x_C, incx, y_C, incy, result_C, lda);
  // CHECK: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: a = (mkl::blas::geru(*handle, N, N, std::complex<double>(alpha_Z.x(),alpha_Z.y()), (std::complex<double>*)x_Z, incx, (std::complex<double>*)y_Z, incy, (std::complex<double>*)result_Z, lda).wait(), 0);
  a = cublasZgeru(handle, N, N, &alpha_Z, x_Z, incx, y_Z, incy, result_Z, lda);
  // CHECK:mkl::blas::gerc(*handle, N, N, std::complex<double>(alpha_Z.x(),alpha_Z.y()), (std::complex<double>*)x_Z, incx, (std::complex<double>*)y_Z, incy, (std::complex<double>*)result_Z, lda).wait();
  cublasZgerc(handle, N, N, &alpha_Z, x_Z, incx, y_Z, incy, result_Z, lda);








  //level 3

  //CHECK:/*
  //CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:a = (mkl::blas::gemm(*handle, trans0==2 ? mkl::transpose::conjtrans : (mkl::transpose)trans0, trans1==2 ? mkl::transpose::conjtrans : (mkl::transpose)trans1, N, N, N, alpha_S, d_A_S, N, d_B_S, N, beta_S, d_C_S, N).wait(), 0);
  a = cublasSgemm(handle, (cublasOperation_t)trans0, (cublasOperation_t)trans1, N, N, N, &alpha_S, d_A_S, N, d_B_S, N, &beta_S, d_C_S, N);
  //CHECK:mkl::blas::gemm(*handle, trans0==2 ? mkl::transpose::conjtrans : (mkl::transpose)trans0, trans1==2 ? mkl::transpose::conjtrans : (mkl::transpose)trans1, N, N, N, alpha_D, d_A_D, N, d_B_D, N, beta_D, d_C_D, N).wait();
  cublasDgemm(handle, (cublasOperation_t)trans0, (cublasOperation_t)trans1, N, N, N, &alpha_D, d_A_D, N, d_B_D, N, &beta_D, d_C_D, N);
  //CHECK:/*
  //CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:a = (mkl::blas::gemm(*handle, mkl::transpose::nontrans, mkl::transpose::nontrans, N, N, N, std::complex<float>(alpha_C.x(),alpha_C.y()), (std::complex<float>*)d_A_C, N, (std::complex<float>*)d_B_C, N, std::complex<float>(beta_C.x(),beta_C.y()), (std::complex<float>*)d_C_C, N).wait(), 0);
  a = cublasCgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha_C, d_A_C, N, d_B_C, N, &beta_C, d_C_C, N);
  //CHECK:mkl::blas::gemm(*handle, mkl::transpose::nontrans, mkl::transpose::nontrans, N, N, N, std::complex<double>(alpha_Z.x(),alpha_Z.y()), (std::complex<double>*)d_A_Z, N, (std::complex<double>*)d_B_Z, N, std::complex<double>(beta_Z.x(),beta_Z.y()), (std::complex<double>*)d_C_Z, N).wait();
  cublasZgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha_Z, d_A_Z, N, d_B_Z, N, &beta_Z, d_C_Z, N);


  //CHECK:dpct::matrix_mem_copy(d_C_S, d_B_S, N, N, N, N, dpct::device_to_device, *handle);
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:a = (mkl::blas::trmm(*handle, (mkl::side)side0, fill0==0 ? mkl::uplo::lower : mkl::uplo::upper, trans0==2 ? mkl::transpose::conjtrans : (mkl::transpose)trans0, (mkl::diag)diag0, N, N, alpha_S, d_A_S, N, d_C_S, N).wait(), 0);
  a = cublasStrmm(handle, (cublasSideMode_t)side0, (cublasFillMode_t)fill0, (cublasOperation_t)trans0, (cublasDiagType_t)diag0, N, N, &alpha_S, d_A_S, N, d_B_S, N, d_C_S, N);
  //CHECK:dpct::matrix_mem_copy(d_C_D, d_B_D, N, N, N, N, dpct::device_to_device, *handle);
  //CHECK-NEXT:mkl::blas::trmm(*handle, (mkl::side)side0, fill0==0 ? mkl::uplo::lower : mkl::uplo::upper, trans0==2 ? mkl::transpose::conjtrans : (mkl::transpose)trans0, (mkl::diag)diag0, N, N, alpha_D, d_A_D, N, d_C_D, N).wait();
  cublasDtrmm(handle, (cublasSideMode_t)side0, (cublasFillMode_t)fill0, (cublasOperation_t)trans0, (cublasDiagType_t)diag0, N, N, &alpha_D, d_A_D, N, d_B_D, N, d_C_D, N);
  //CHECK:dpct::matrix_mem_copy(d_C_C, d_B_C, N, N, N, N, dpct::device_to_device, *handle);
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:a = (mkl::blas::trmm(*handle, mkl::side::left, mkl::uplo::lower, mkl::transpose::nontrans, mkl::diag::unit, N, N, std::complex<float>(alpha_C.x(),alpha_C.y()), (std::complex<float>*)d_A_C, N, (std::complex<float>*)d_C_C, N).wait(), 0);
  a = cublasCtrmm(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_UNIT, N, N, &alpha_C, d_A_C, N, d_B_C, N, d_C_C, N);
  //CHECK:dpct::matrix_mem_copy(d_C_Z, d_B_Z, N, N, N, N, dpct::device_to_device, *handle);
  //CHECK-NEXT:mkl::blas::trmm(*handle, mkl::side::left, mkl::uplo::lower, mkl::transpose::nontrans, mkl::diag::unit, N, N, std::complex<double>(alpha_Z.x(),alpha_Z.y()), (std::complex<double>*)d_A_Z, N, (std::complex<double>*)d_C_Z, N).wait();
  cublasZtrmm(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_UNIT, N, N, &alpha_Z, d_A_Z, N, d_B_Z, N, d_C_Z, N);


  //CHECK:a = (mkl::blas::gemmt(*handle, fill0==0 ? mkl::uplo::lower : mkl::uplo::upper, trans1==2 ? mkl::transpose::conjtrans : (mkl::transpose)trans1, trans1==0 ? mkl::transpose::trans : mkl::transpose::nontrans, N, N, alpha_S, d_A_S, N, d_B_S, N, beta_S, d_C_S, N).wait(), 0);
  a = cublasSsyrkx(handle, (cublasFillMode_t)fill0, (cublasOperation_t)trans1, N, N, &alpha_S, d_A_S, N, d_B_S, N, &beta_S, d_C_S, N);
  //CHECK:mkl::blas::gemmt(*handle, fill0==0 ? mkl::uplo::lower : mkl::uplo::upper, trans1==2 ? mkl::transpose::conjtrans : (mkl::transpose)trans1, trans1==0 ? mkl::transpose::trans : mkl::transpose::nontrans, N, N, alpha_D, d_A_D, N, d_B_D, N, beta_D, d_C_D, N).wait();
  cublasDsyrkx(handle, (cublasFillMode_t)fill0, (cublasOperation_t)trans1, N, N, &alpha_D, d_A_D, N, d_B_D, N, &beta_D, d_C_D, N);



  // CHECK: dpct::matrix_mem_copy(d_C_S, d_B_S, N, N, N, N, dpct::device_to_device, *handle);
  // CHECK-NEXT: mkl::blas::trmm(*handle, (mkl::side)side0, fill0==0 ? mkl::uplo::lower : mkl::uplo::upper, trans0==2 ? mkl::transpose::conjtrans : (mkl::transpose)trans0, (mkl::diag)diag0, N, N, alpha_S, d_A_S, N, d_C_S, N).wait();
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1041:{{[0-9]+}}: SYCL uses exceptions to report errors and does not use the error codes. 0 is used in if statement. You need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: if(int stat = 0){}
  if(int stat = cublasStrmm(handle, (cublasSideMode_t)side0, (cublasFillMode_t)fill0, (cublasOperation_t)trans0, (cublasDiagType_t)diag0, N, N, &alpha_S, d_A_S, N, d_B_S, N, d_C_S, N)){}

  // CHECK: /*
  // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  // CHECK-NEXT: */
  // CHECK-NEXT: if(int stat = (mkl::blas::gemm(*handle, trans0==2 ? mkl::transpose::conjtrans : (mkl::transpose)trans0, trans1==2 ? mkl::transpose::conjtrans : (mkl::transpose)trans1, N, N, N, alpha_S, d_A_S, N, d_B_S, N, beta_S, d_C_S, N).wait(), 0)){}
  if(int stat = cublasSgemm(handle, (cublasOperation_t)trans0, (cublasOperation_t)trans1, N, N, N, &alpha_S, d_A_S, N, d_B_S, N, &beta_S, d_C_S, N)){}


}

// CHECK: int foo1() try {
// CHECK-NEXT:   dpct::matrix_mem_copy(d_C_S, d_B_S, N, N, N, N, dpct::device_to_device, *handle);
// CHECK-NEXT:   mkl::blas::trmm(*handle, (mkl::side)side0, fill0==0 ? mkl::uplo::lower : mkl::uplo::upper, trans0==2 ? mkl::transpose::conjtrans : (mkl::transpose)trans0, (mkl::diag)diag0, N, N, alpha_S, d_A_S, N, d_C_S, N).wait();
// CHECK-NEXT:   /*
// CHECK-NEXT:   DPCT1041:{{[0-9]+}}: SYCL uses exceptions to report errors and does not use the error codes. 0 is used in return statement. You need to rewrite this code.
// CHECK-NEXT:   */
// CHECK-NEXT:   return 0;
// CHECK-NEXT: }
int foo1(){
  return cublasStrmm(handle, (cublasSideMode_t)side0, (cublasFillMode_t)fill0, (cublasOperation_t)trans0, (cublasDiagType_t)diag0, N, N, &alpha_S, d_A_S, N, d_B_S, N, d_C_S, N);
}

// CHECK:int foo2() try {
// CHECK-NEXT:  /*
// CHECK-NEXT:  DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
// CHECK-NEXT:  */
// CHECK-NEXT:  return (mkl::blas::gemm(*handle, trans0==2 ? mkl::transpose::conjtrans : (mkl::transpose)trans0, trans1==2 ? mkl::transpose::conjtrans : (mkl::transpose)trans1, N, N, N, alpha_S, d_A_S, N, d_B_S, N, beta_S, d_C_S, N).wait(), 0);
// CHECK-NEXT:}
int foo2(){
  return cublasSgemm(handle, (cublasOperation_t)trans0, (cublasOperation_t)trans1, N, N, N, &alpha_S, d_A_S, N, d_B_S, N, &beta_S, d_C_S, N);
}