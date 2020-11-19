// RUN: dpct --format-range=none -out-root %T/builtin_warpSize %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/builtin_warpSize/builtin_warpSize.dp.cpp --match-full-lines %s

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>


__global__ void foo(){
  // CHECK: int a = item_ct1.get_sub_group().get_local_range().get(0);
  // CHECK-NEXT: int warpSize = 1;
  // CHECK-NEXT: warpSize = 2;
  // CHECK-NEXT: int c= warpSize;
  int a = warpSize;
  int warpSize = 1;
  warpSize = 2;
  int c= warpSize;
}

