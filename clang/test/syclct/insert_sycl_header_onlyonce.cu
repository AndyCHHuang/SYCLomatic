// RUN: syclct -out-root %T %s -- -x cuda --cuda-host-only --cuda-path="%cuda-path"
// RUN: FileCheck --input-file %T/insert_sycl_header_onlyonce.dp.cpp --match-full-lines %s

// CHECK: #include <CL/sycl.hpp>
// CHECK-NEXT: #include <dpct/dpct.hpp>
// CHECK: #include <stdio.h>
// CHECK-NOT:#include <CL/sycl.hpp>
#include <stdio.h>
#include <cuda.h>

int main(){
  return 0;
}

