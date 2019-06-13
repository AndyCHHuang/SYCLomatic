//===--- MapNames.cpp -----------------------------------*- C++ -*---===//
//
// Copyright (C) 2018 Intel Corporation. All rights reserved.
//
// The information and source code contained herein is the exclusive
// property of Intel Corporation and may not be disclosed, examined
// or reproduced in whole or in part without explicit written authorization
// from the company.
//
//===-----------------------------------------------------------------===//

#ifndef __CLANG_SYCLCT_MATH_H__
#define __CLANG_SYCLCT_MATH_H__

#define __SYCLCT__
#ifdef __SYCLCT__

float max(float a, float b);
int  min(int, int);

#if (defined(_WIN64) || defined(_WIN32))

#else
#if !defined(_GLIBCXX_RELEASE) || defined(_GLIBCXX_RELEASE) && _GLIBCXX_RELEASE < 7
int signbit(float x);
int signbit(double x);
int signbit(long double x);
int isfinite(float x);
int isfinite(double x);
int isfinite(long double x);
float modf(float a, float *b);
#endif
#endif

float powif(float a, int b);
double powi(double a, int b);
float norm3d(float a, float b, float c);
float norm4d(float a, float b, float c, float d);
void sincos(float a, float *sptr, float *cptr);
float cyl_bessel_i0(float a);
float cyl_bessel_i1(float a);

__host__ double atomicAdd(double* address, double val);
__host__ unsigned long long atomicAdd(unsigned long long* address, unsigned long long val);
__host__ long atomicAdd(long* address, long val);
__host__ unsigned long atomicAdd(unsigned long* address, unsigned long val);
__host__ int atomicAdd(int* address, int val);
__host__ unsigned int atomicAdd(unsigned int* address, unsigned int val);



/// ---Fix for Window CUDA10.1--------------------------------
#if CUDA_VERSION >= 10000 && (defined(_WIN64) || defined(_WIN32))
extern __host__ __device__ unsigned cudaConfigureCall(dim3 gridDim, dim3 blockDim,
  size_t sharedMem = 0, void *stream = 0);
#endif


/// ---Fix for Windows CUDA >=9---------------------------------
#if (CUDA_VERSION >= 9000) && (defined(_WIN64) || defined(_WIN32))
__device__ float roundf(float __a);
extern __device__  int __finitel(long double) ;
extern __device__ __device_builtin__ int  __isinfl(long double) ;
extern __device__ __device_builtin__ int  __isnanl(long double) ;
#endif


/// Fix issue:va_printf is not defined in _CubLog.
/// solution: for migration, just define _CubLog to empty.
#define _CubLog(format, ...)


#endif
#endif
