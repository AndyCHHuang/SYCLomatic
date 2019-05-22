//===--- thrust_test-pennet_simple_pstl.cpp---------------------*- C++
//-*---===//
//
// Copyright (C) 2018 Intel Corporation. All rights reserved.
//
// The information and source code contained herein is the exclusive
// property of Intel Corporation and may not be disclosed, examined
// or reproduced in whole or in part without explicit written authorization
// from the company.
//
//===-----------------------------------------------------------------===//
/*
Thrust test case:
package required:
install tbb and pstl library(eg. install intel parallel studio)
cd /path/to/tbb/bin &&  source  tbbvars.sh intel64
cd /path/to/pstl/bin && source   pstlvars.sh intel64 auto_pstlroot

build:
compute++   thrust_test-pennet_simple_pstl.cpp.cpp
  -I/Path/to/ComputeCpp-CE-1.0.2-Ubuntu-16.04-x86_64/include/
  -std=c++11
  -L/Path/to/ComputeCpp-CE-1.0.2-Ubuntu-16.04-x86_64/lib
  -lComputeCpp   -I/path/to/syclct-install/include/
  -sycl-driver
*/
#include <algorithm>
#include <cstdio>

#include <CL/sycl.hpp>
#include <syclct/syclct.hpp>
#include <syclct/syclct_thrust.hpp>

int main() {

  int *mapsp1D, *mapspkeyD, *mapspvalD;
  int numsH = 10;

  // cudaMalloc(&mapsp1D, numsH*sizeof(int));
  // cudaMalloc(&mapspkeyD, numsH*sizeof(int));
  // cudaMalloc(&mapspvalD, numsH*sizeof(int));
  syclct::sycl_malloc((void **)&mapsp1D, numsH * sizeof(int));
  syclct::sycl_malloc((void **)&mapspkeyD, numsH * sizeof(int));
  syclct::sycl_malloc((void **)&mapspvalD, numsH * sizeof(int));

  thrust::device_ptr<int> mapsp1T(mapsp1D);
  thrust::device_ptr<int> mapspkeyT(mapspkeyD);
  thrust::device_ptr<int> mapspvalT(mapspvalD);

  mapspkeyT[0] = 100;
  mapspkeyT[1] = 101;
  mapspkeyT[2] = 102;
  mapspkeyT[3] = 103;
  mapspkeyT[4] = 104;
  mapspkeyT[5] = 105;
  mapspkeyT[6] = 106;
  mapspkeyT[7] = 107;
  mapspkeyT[8] = 108;
  mapspkeyT[9] = 109;

  thrust::copy(mapsp1T, mapsp1T + numsH, mapspkeyT);
  thrust::sequence(mapspvalT, mapspvalT + numsH);
  thrust::stable_sort_by_key(mapspkeyT, mapspkeyT + numsH, mapspvalT);

  for (int i = 0; i < numsH; ++i) {
    std::cout << "i = " << i << ", " << mapspkeyT[i] << " " << mapspvalT[i]
              << ";\n";
  }
  std::cout << std::endl << "done" << std::endl;
}
