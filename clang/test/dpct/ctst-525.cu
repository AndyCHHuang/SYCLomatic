// RUN: dpct --format-range=none -out-root %T/ctst-525 %s --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/ctst-525/ctst-525.dp.cpp --match-full-lines %s
#include <cuda_runtime.h>
class C {
  int nDevices;
public:
  void problem() {
    // CHECK: nDevices = dpct::dev_mgr::instance().device_count();
    cudaGetDeviceCount(&nDevices);
  }
};

