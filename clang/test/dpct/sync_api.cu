// UNSUPPORTED: cuda-8.0
// UNSUPPORTED: v8.0
// RUN: dpct --format-range=none -out-root %T/sync_api %s --cuda-include-path="%cuda-path/include" --use-experimental-features=nd_range_barrier -- -x cuda --cuda-host-only
// RUN: FileCheck %s --match-full-lines --input-file %T/sync_api/sync_api.dp.cpp

// CHECK: #include <CL/sycl.hpp>
// CHECK-NEXT: #include <dpct/dpct.hpp>
#include "cooperative_groups.h"
namespace cg = cooperative_groups;
using namespace cooperative_groups;

// CHECK: #define TB(b) auto b = item_ct1.get_group();
#define TB(b) cg::thread_block b = cg::this_thread_block();

__device__ void foo(int i) {}

#define FOO(x) foo(x)

// CHECK: void k(sycl::nd_item<3> item_ct1) {
__global__ void k() {
  // CHECK: auto cta = item_ct1.get_group();
  cg::thread_block cta = cg::this_thread_block();
  // CHECK: item_ct1.barrier();
  cg::sync(cta);

  // CHECK: auto block = item_ct1.get_group();
  cg::thread_block block = cg::this_thread_block();
  // CHECK: item_ct1.barrier();
  __syncthreads();
  // CHECK: item_ct1.barrier();
  block.sync();
  // CHECK: item_ct1.barrier();
  cg::sync(block);
  // CHECK: item_ct1.barrier();
  cg::this_thread_block().sync();
  // CHECK: item_ct1.barrier();
  cg::sync(cg::this_thread_block());

  // CHECK: auto b0 = item_ct1.get_group(), b1 = item_ct1.get_group();
  cg::thread_block b0 = cg::this_thread_block(), b1 = cg::this_thread_block();

  TB(blk);

  int p;
  // CHECK: /*
  // CHECK-NEXT: DPCT1078:{{[0-9]+}}: Consider replacing memory_order::acq_rel with memory_order::seq_cst for correctness if strong memory order restrictions are needed.
  // CHECK-NEXT: */
  // CHECK-NEXT: sycl::ext::oneapi::atomic_fence(sycl::ext::oneapi::memory_order::acq_rel, sycl::ext::oneapi::memory_scope::work_group);
  __threadfence_block();
  // CHECK: /*
  // CHECK-NEXT: DPCT1078:{{[0-9]+}}: Consider replacing memory_order::acq_rel with memory_order::seq_cst for correctness if strong memory order restrictions are needed.
  // CHECK-NEXT: */
  // CHECK-NEXT: sycl::ext::oneapi::atomic_fence(sycl::ext::oneapi::memory_order::acq_rel, sycl::ext::oneapi::memory_scope::device);
  __threadfence();
  // CHECK: /*
  // CHECK-NEXT: DPCT1078:{{[0-9]+}}: Consider replacing memory_order::acq_rel with memory_order::seq_cst for correctness if strong memory order restrictions are needed.
  // CHECK-NEXT: */
  // CHECK-NEXT: sycl::ext::oneapi::atomic_fence(sycl::ext::oneapi::memory_order::acq_rel, sycl::ext::oneapi::memory_scope::system);
  __threadfence_system();
  // CHECK: item_ct1.barrier();
  // CHECK-NEXT: sycl::all_of_group(item_ct1.get_group(), p);
  __syncthreads_and(p);
  // CHECK: item_ct1.barrier();
  // CHECK-NEXT: sycl::any_of_group(item_ct1.get_group(), p);
  __syncthreads_or(p);
  // CHECK: item_ct1.barrier();
  // CHECK-NEXT: sycl::reduce_over_group(item_ct1.get_group(), p == 0 ? 0 : 1, sycl::ext::oneapi::plus<>());
  __syncthreads_count(p);
  // CHECK: item_ct1.barrier();
  __syncwarp(0xffffffff);

  // CHECK: int a = (item_ct1.barrier(), sycl::all_of_group(item_ct1.get_group(), p));
  int a = __syncthreads_and(p);
  // CHECK: int b = (item_ct1.barrier(), sycl::any_of_group(item_ct1.get_group(), p));
  int b = __syncthreads_or(p);
  // CHECK: int c = (item_ct1.barrier(), sycl::reduce_over_group(item_ct1.get_group(), p == 0 ? 0 : 1, sycl::ext::oneapi::plus<>()));
  int c = __syncthreads_count(p);

  // CHECK: foo((item_ct1.barrier(), sycl::all_of_group(item_ct1.get_group(), p)));
  foo(__syncthreads_and(p));
  // CHECK: foo((item_ct1.barrier(), sycl::any_of_group(item_ct1.get_group(), p)));
  foo(__syncthreads_or(p));
  // CHECK: foo((item_ct1.barrier(), sycl::reduce_over_group(item_ct1.get_group(), p == 0 ? 0 : 1, sycl::ext::oneapi::plus<>())));
  foo(__syncthreads_count(p));

  // CHECK: FOO((item_ct1.barrier(), sycl::all_of_group(item_ct1.get_group(), p)));
  FOO(__syncthreads_and(p));
  // CHECK: FOO((item_ct1.barrier(), sycl::any_of_group(item_ct1.get_group(), p)));
  FOO(__syncthreads_or(p));
  // CHECK: FOO((item_ct1.barrier(), sycl::reduce_over_group(item_ct1.get_group(), p == 0 ? 0 : 1, sycl::ext::oneapi::plus<>())));
  FOO(__syncthreads_count(p));
}

// CHECK: void kernel(sycl::nd_item<3> item_ct1,
// CHECK-NEXT:            sycl::ext::oneapi::atomic_ref<unsigned int,sycl::ext::oneapi::memory_order::seq_cst,sycl::ext::oneapi::memory_scope::device,sycl::access::address_space::global_space> &sync_ct1) {
// CHECK-NEXT:  dpct::experimental::nd_range_barrier(item_ct1, sync_ct1);
// CHECK-NEXT:}
__global__ void kernel() {
  cg::grid_group grid = cg::this_grid();
  grid.sync();
}

int main() {
// CHECK:  {
// CHECK-NEXT:    dpct::global_memory<unsigned int, 0> d_sync_ct1(0);
// CHECK-NEXT:    unsigned *sync_ct1 = d_sync_ct1.get_ptr(dpct::get_default_queue());
// CHECK-NEXT:    dpct::get_default_queue().memset(sync_ct1, 0, sizeof(int)).wait();
// CHECK-NEXT:    dpct::get_default_queue().parallel_for(
// CHECK-NEXT:      sycl::nd_range<3>(sycl::range<3>(1, 1, 2) * sycl::range<3>(1, 1, 2), sycl::range<3>(1, 1, 2)), 
// CHECK-NEXT:      [=](sycl::nd_item<3> item_ct1)  {
// CHECK-NEXT:        auto atm_sync_ct1 = sycl::ext::oneapi::atomic_ref<unsigned int,sycl::ext::oneapi::memory_order::seq_cst,sycl::ext::oneapi::memory_scope::device,sycl::access::address_space::global_space>(sync_ct1[0]);
// CHECK-NEXT:        kernel(item_ct1, atm_sync_ct1);
// CHECK-NEXT:      }).wait();
// CHECK-NEXT:  }
  kernel<<<2, 2>>>();
  cudaDeviceSynchronize();
  return 0;
}