// RUN: dpct --format-range=none --usm-level=none -out-root %T %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only -D__NVCC__ -D__CUDACC__
// RUN: FileCheck --input-file %T/predefined_macro_replacement.dp.cpp --match-full-lines %s
#include <stdio.h>
//CHECK: #ifdef DPCPP_COMPATIBILITY_TEMP
//CHECK-NEXT: void hello(sycl::stream [[STREAM:stream_ct1]]) { [[STREAM]] << "foo"; }
#ifdef __CUDA_ARCH__
__global__ void hello() { printf("foo"); }
#else
__global__ void hello() { printf("other"); }
#endif

//CHECK: #ifndef DPCPP_COMPATIBILITY_TEMP
#ifndef __NVCC__
__global__ void hello2() { printf("hello2"); }
#endif
//CHECK: #if defined(CL_SYCL_LANGUAGE_VERSION)
#if defined(__CUDACC__)
__global__ void hello3() { printf("hello2"); }
#endif

#if defined(xxx)
__global__ void hello4() { printf("hello2"); }
//CHECK: #elif defined(DPCPP_COMPATIBILITY_TEMP)
//CHECK-NEXT: void hello4(sycl::stream [[STREAM]]) { [[STREAM]] << "hello2"; }
#elif defined(__CUDA_ARCH__)
__global__ void hello4() { printf("hello2"); }
#endif

#if defined(xxx)
__global__ void hello5() { printf("hello2"); }
//CHECK: #elif (DPCPP_COMPATIBILITY_TEMP >= 400)
//CHECK-NEXT: void hello5(sycl::stream [[STREAM]]) { [[STREAM]] << "hello2"; }
#elif (__CUDA_ARCH__ >= 400)
__global__ void hello5() { printf("hello2"); }
#endif

//CHECK: #if defined(DPCPP_COMPATIBILITY_TEMP)
//CHECK-NEXT: void hello6(sycl::stream [[STREAM]]) { [[STREAM]] << "hello2"; }
#if defined(__CUDA_ARCH__)
__global__ void hello6() { printf("hello2"); }
#endif

//CHECK: #ifndef DPCPP_COMPATIBILITY_TEMP
//CHECK-NEXT: __global__ void hello7() { printf("hello2"); }
//CHECK-NEXT: #else
//CHECK-NEXT: void hello7(sycl::stream [[STREAM]]) { [[STREAM]] << "hello2"; }
#ifndef __CUDA_ARCH__
__global__ void hello7() { printf("hello2"); }
#else
__global__ void hello7() { printf("hello2"); }
#endif

__global__ void test(){
//CHECK:#if (DPCPP_COMPATIBILITY_TEMP >= 400) &&  (DPCPP_COMPATIBILITY_TEMP >= 400)
//CHECK-NEXT:[[STREAM]] << ">400, \n";
//CHECK-NEXT:#elif (DPCPP_COMPATIBILITY_TEMP >200)
//CHECK-NEXT:printf(">200, \n");
//CHECK-NEXT:#else
//CHECK-NEXT:printf("<200, \n");
//CHECK-NEXT:#endif
#if (__CUDA_ARCH__ >= 400) &&  (__CUDA_ARCH__ >= 400)
printf(">400, \n");
#elif (__CUDA_ARCH__ >200)
printf(">200, \n");
#else
printf("<200, \n");
#endif
}


int main() {
//CHECK: #if defined(DPCPP_COMPATIBILITY_TEMP)
//CHECK-NEXT:     dpct::get_default_queue().submit(
#if defined(__NVCC__)
  hello<<<1,1>>>();
#else
  hello();
#endif
  return 0;
}

//CHECK: #define AAA DPCPP_COMPATIBILITY_TEMP
//CHECK-NEXT: #define BBB CL_SYCL_LANGUAGE_VERSION
//CHECK-NEXT: #define CCC DPCPP_COMPATIBILITY_TEMP
#define AAA __CUDA_ARCH__
#define BBB __CUDACC__
#define CCC __NVCC__
