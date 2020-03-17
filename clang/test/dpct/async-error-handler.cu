// RUN: dpct --usm-level=none -out-root %T %s --cuda-include-path="%cuda-path/include" --sycl-named-lambda --always-use-async-handler -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/async-error-handler.dp.cpp --match-full-lines %s


int main() {
  // CHECK: sycl::queue *s0, *s1, *s2;
  cudaStream_t s0, s1, s2;

  // CHECK: s0 = dpct::get_current_device().create_queue(true);
  cudaStreamCreate(&s0);

  // CHECK: /*
  // CHECK-NEXT: DPCT1025:{{[0-9]+}}: The SYCL queue is created ignoring the flag/priority options.
  // CHECK-NEXT: */
  // CHECK-NEXT: s1 = dpct::get_current_device().create_queue(true);
  cudaStreamCreateWithFlags(&s1, cudaStreamDefault);

  // CHECK: /*
  // CHECK-NEXT: DPCT1025:{{[0-9]+}}: The SYCL queue is created ignoring the flag/priority options.
  // CHECK-NEXT: */
  // CHECK-NEXT: s2 = dpct::get_current_device().create_queue(true);
  cudaStreamCreateWithPriority(&s2, cudaStreamDefault, 2);
}
