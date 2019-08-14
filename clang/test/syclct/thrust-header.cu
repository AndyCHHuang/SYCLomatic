// UNSUPPORTED: cuda-8.0
// RUN: syclct -out-root %T %s  -- -x cuda --cuda-host-only --cuda-path="%cuda-path"
// RUN: FileCheck --input-file %T/thrust-header.dp.cpp --match-full-lines %s
// CHECK: #include <CL/sycl.hpp>
// CHECK-NEXT: #include <dpct/dpct.hpp>
// CHECK-NEXT: #include <cstdio>
// CHECK-NEXT: #include <algorithm>
#include <cstdio>
#include <algorithm>
// CHECK: #include <dpstd/algorithm>
// CHECK-NEXT: #include <dpstd/execution>
// CHECK-NEXT: #include <dpct/dpct_dpstd_utils.hpp>
#include <cuda_runtime.h>
#include <thrust/copy.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
int main() {

  int *mapsp1D, *mapspkeyD,*mapspvalD;
  int numsH=10;

  cudaMalloc(&mapsp1D, numsH*sizeof(int));
  cudaMalloc(&mapspkeyD, numsH*sizeof(int));
  cudaMalloc(&mapspvalD, numsH*sizeof(int));

// CHECK:  dpct::device_ptr<int> mapsp1T(mapsp1D);
  thrust::device_ptr<int> mapsp1T(mapsp1D);
// CHECK:  dpct::device_ptr<int> mapspkeyT(mapspkeyD);
  thrust::device_ptr<int> mapspkeyT(mapspkeyD);
// CHECK:  dpct::device_ptr<int> mapspvalT(mapspvalD);
  thrust::device_ptr<int> mapspvalT(mapspvalD);

// CHECK:  std::copy(dpstd::execution::make_sycl_policy<class Policy_{{[0-9a-f]+}}>(dpstd::execution::sycl), mapsp1T, mapsp1T + numsH, mapspkeyT);
  thrust::copy(mapsp1T, mapsp1T + numsH, mapspkeyT);
// CHECK:  dpct::sequence(dpstd::execution::make_sycl_policy<class Policy_{{[0-9a-f]+}}>(dpstd::execution::sycl), mapspvalT, mapspvalT + numsH);
  thrust::sequence(mapspvalT, mapspvalT + numsH);
// CHECK:  dpct::stable_sort_by_key(dpstd::execution::make_sycl_policy<class Policy_{{[0-9a-f]+}}>(dpstd::execution::sycl), mapspkeyT, mapspkeyT + numsH, mapspvalT);
  thrust::stable_sort_by_key(mapspkeyT, mapspkeyT + numsH, mapspvalT);
}
