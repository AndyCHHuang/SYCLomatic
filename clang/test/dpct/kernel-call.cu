// RUN: dpct --format-range=none --no-cl-namespace-inline --usm-level=none -out-root %T %s --cuda-include-path="%cuda-path/include" --sycl-named-lambda -- -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/kernel-call.dp.cpp --match-full-lines %s

#include <stdio.h>

// CHECK: void testKernel(int L, int M, int N,
// CHECK-NEXT: cl::sycl::nd_item<3> [[ITEMNAME:item_ct1]]);
__global__ void testKernel(int L, int M, int N);

// CHECK: void testKernelPtr(const int *L, const int *M, int N,
// CHECK-NEXT: cl::sycl::nd_item<3> [[ITEMNAME:item_ct1]]) {
__global__ void testKernelPtr(const int *L, const int *M, int N) {
  // CHECK: int gtid = [[ITEMNAME]].get_group(2) * [[ITEMNAME]].get_local_range().get(2) + [[ITEMNAME]].get_local_id(2);
  int gtid = blockIdx.x * blockDim.x + threadIdx.x;
}


// CHECK: // Test Launch Bounds
// CHECK-NEXT: void testKernel(int L, int M, int N,
// CHECK-NEXT:                 cl::sycl::nd_item<3> [[ITEMNAME:item_ct1]]) {
__launch_bounds__(256, 512) // Test Launch Bounds
__global__ void testKernel(int L, int M, int N) {
  // CHECK: int gtid = [[ITEMNAME]].get_group(2) * [[ITEMNAME]].get_local_range().get(2) + [[ITEMNAME]].get_local_id(2);
  int gtid = blockIdx.x * blockDim.x + threadIdx.x;
}

// CHECK: void helloFromGPU(int i, cl::sycl::nd_item<3> item_ct1) {
// CHECK-NEXT:     int a = item_ct1.get_group(2) * item_ct1.get_local_range().get(2) + item_ct1.get_local_id(2) + item_ct1.get_group(2) +
// CHECK-NEXT:     item_ct1.get_local_range().get(2) + item_ct1.get_local_id(2);
// CHECK-NEXT: }
__global__ void helloFromGPU(int i) {
  int a = blockIdx.x * blockDim.x + threadIdx.x + blockIdx.x +
          blockDim.x + threadIdx.x;
}

// CHECK: void helloFromGPU(cl::sycl::nd_item<3> item_ct1) {
// CHECK-NEXT:     int a = item_ct1.get_group(2) * item_ct1.get_local_range().get(2) + item_ct1.get_local_id(2) + item_ct1.get_group(2) +
// CHECK-NEXT:     item_ct1.get_local_range().get(2) + item_ct1.get_local_id(2);
// CHECK-NEXT: }
__global__ void helloFromGPU(void) {
  int a = blockIdx.x * blockDim.x + threadIdx.x + blockIdx.x +
          blockDim.x + threadIdx.x;
}

// CHECK: void helloFromGPU2(cl::sycl::nd_item<3> item_ct1) {
// CHECK-NEXT:     int a = item_ct1.get_group(2) * item_ct1.get_local_range().get(2) + item_ct1.get_local_id(2) + item_ct1.get_group(2) +
// CHECK-NEXT:     item_ct1.get_local_range().get(2) + item_ct1.get_local_id(2);
// CHECK-NEXT: }
__global__ void helloFromGPU2() {
  int a = blockIdx.x * blockDim.x + threadIdx.x + blockIdx.x +
          blockDim.x + threadIdx.x;
}

void testReference(const int &i) {
  dim3 griddim = 2;
  dim3 threaddim = 32;
  // CHECK:   dpct::get_default_queue().submit(
  // CHECK-NEXT:     [&](cl::sycl::handler &cgh) {
  // CHECK-NEXT:       auto dpct_global_range = griddim * threaddim;
  // CHECK-EMPTY:
  // CHECK-NEXT:       cgh.parallel_for<dpct_kernel_name<class helloFromGPU_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:         cl::sycl::nd_range<3>(cl::sycl::range<3>(dpct_global_range.get(2), dpct_global_range.get(1), dpct_global_range.get(0)), cl::sycl::range<3>(threaddim.get(2), threaddim.get(1), threaddim.get(0))),
  // CHECK-NEXT:         [=](cl::sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:           helloFromGPU(i, item_ct1);
  // CHECK-NEXT:         });
  // CHECK-NEXT:     });
  helloFromGPU<<<griddim, threaddim>>>(i);

}

struct TestThis {
  struct TestMember {
    int arg1, arg2;
  } args;
  int arg3;
  dim3 griddim, threaddim;
  void test() {
    /// Kernel function is called in method declaration, and fields are used as arguments.
    /// Check the miggration of implicit "this" pointer.
    // CHECK:  dpct::get_default_queue().submit(
    // CHECK-NEXT:     [&](cl::sycl::handler &cgh) {
    // CHECK-NEXT:       auto dpct_global_range = griddim * threaddim;
    // CHECK-EMPTY:
    // CHECK-NEXT:       auto args_arg1_ct0 = args.arg1;
    // CHECK-NEXT:       auto args_arg2_ct1 = args.arg2;
    // CHECK-NEXT:       auto arg3_ct2 = arg3;
    // CHECK-EMPTY:
    // CHECK-NEXT:       cgh.parallel_for<dpct_kernel_name<class testKernel_{{[a-f0-9]+}}>>(
    // CHECK-NEXT:         cl::sycl::nd_range<3>(cl::sycl::range<3>(dpct_global_range.get(2), dpct_global_range.get(1), dpct_global_range.get(0)), cl::sycl::range<3>(threaddim.get(2), threaddim.get(1), threaddim.get(0))),
    // CHECK-NEXT:         [=](cl::sycl::nd_item<3> item_ct1) {
    // CHECK-NEXT:           testKernel(args_arg1_ct0, args_arg2_ct1, arg3_ct2, item_ct1);
    // CHECK-NEXT:         });
    // CHECK-NEXT:     });
    testKernel<<<griddim, threaddim>>>(args.arg1, args.arg2, arg3);
  }
};

int arr[16];

int main() {
  dim3 griddim = 2;
  dim3 threaddim = 32;
  void *karg1 = 0;
  const int *karg2 = 0;
  int karg3 = 80;
  // CHECK: {
  // CHECK-NEXT:   std::pair<dpct::buffer_t, size_t> karg1_buf_ct0 = dpct::get_buffer_and_offset((const int *)karg1);
  // CHECK-NEXT:   size_t karg1_offset_ct0 = karg1_buf_ct0.second;
  // CHECK-NEXT:   std::pair<dpct::buffer_t, size_t> karg2_buf_ct1 = dpct::get_buffer_and_offset(karg2);
  // CHECK-NEXT:   size_t karg2_offset_ct1 = karg2_buf_ct1.second;
  // CHECK-NEXT:   dpct::get_default_queue().submit(
  // CHECK-NEXT:     [&](cl::sycl::handler &cgh) {
  // CHECK-NEXT:       auto karg1_acc_ct0 = karg1_buf_ct0.first.get_access<cl::sycl::access::mode::read_write>(cgh);
  // CHECK-NEXT:       auto karg2_acc_ct1 = karg2_buf_ct1.first.get_access<cl::sycl::access::mode::read_write>(cgh);
  // CHECK-EMPTY:
  // CHECK-NEXT:       auto dpct_global_range = griddim * threaddim;
  // CHECK-EMPTY:
  // CHECK-NEXT:       cgh.parallel_for<dpct_kernel_name<class testKernelPtr_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:         cl::sycl::nd_range<3>(cl::sycl::range<3>(dpct_global_range.get(2), dpct_global_range.get(1), dpct_global_range.get(0)), cl::sycl::range<3>(threaddim.get(2), threaddim.get(1), threaddim.get(0))),
  // CHECK-NEXT:         [=](cl::sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:           const int *karg1_ct0 = (const int *)(&karg1_acc_ct0[0] + karg1_offset_ct0);
  // CHECK-NEXT:           const int *karg2_ct1 = (const int *)(&karg2_acc_ct1[0] + karg2_offset_ct1);
  // CHECK-NEXT:           testKernelPtr(karg1_ct0, karg2_ct1, karg3, item_ct1);
  // CHECK-NEXT:         });
  // CHECK-NEXT:     });
  // CHECK-NEXT: }
  testKernelPtr<<<griddim, threaddim>>>((const int *)karg1, karg2, karg3);

  int karg1int = 1;
  int karg2int = 2;
  int karg3int = 3;
  int intvar = 20;
  // CHECK:   dpct::get_default_queue().submit(
  // CHECK-NEXT:     [&](cl::sycl::handler &cgh) {
  // CHECK-NEXT:       cgh.parallel_for<dpct_kernel_name<class testKernel_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:         cl::sycl::nd_range<3>(cl::sycl::range<3>(1, 1, 10) * cl::sycl::range<3>(1, 1, intvar), cl::sycl::range<3>(1, 1, intvar)),
  // CHECK-NEXT:         [=](cl::sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:           testKernel(karg1int, karg2int, karg3int, item_ct1);
  // CHECK-NEXT:         });
  // CHECK-NEXT:     });
  testKernel<<<10, intvar>>>(karg1int, karg2int, karg3int);

  struct KernelPointer {
    const int *arg1, *arg2;
  } args;
  // CHECK: {
  // CHECK-NEXT:  std::pair<dpct::buffer_t, size_t> args_arg1_buf_ct0 = dpct::get_buffer_and_offset(args.arg1);
  // CHECK-NEXT:  size_t args_arg1_offset_ct0 = args_arg1_buf_ct0.second;
  // CHECK-NEXT:  std::pair<dpct::buffer_t, size_t> args_arg2_buf_ct1 = dpct::get_buffer_and_offset(args.arg2);
  // CHECK-NEXT:  size_t args_arg2_offset_ct1 = args_arg2_buf_ct1.second;
  // CHECK-NEXT:  dpct::get_default_queue().submit(
  // CHECK-NEXT:    [&](cl::sycl::handler &cgh) {
  // CHECK-NEXT:      auto args_arg1_acc_ct0 = args_arg1_buf_ct0.first.get_access<cl::sycl::access::mode::read_write>(cgh);
  // CHECK-NEXT:      auto args_arg2_acc_ct1 = args_arg2_buf_ct1.first.get_access<cl::sycl::access::mode::read_write>(cgh);
  // CHECK-EMPTY:
  // CHECK-NEXT:      cgh.parallel_for<dpct_kernel_name<class testKernelPtr_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:         cl::sycl::nd_range<3>(cl::sycl::range<3>(1, 2, 1), cl::sycl::range<3>(1, 2, 1)),
  // CHECK-NEXT:        [=](cl::sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:          const int *args_arg1_ct0 = (const int *)(&args_arg1_acc_ct0[0] + args_arg1_offset_ct0);
  // CHECK-NEXT:          const int *args_arg2_ct1 = (const int *)(&args_arg2_acc_ct1[0] + args_arg2_offset_ct1);
  // CHECK-NEXT:          testKernelPtr(args_arg1_ct0, args_arg2_ct1, karg3int, item_ct1);
  // CHECK-NEXT:        });
  // CHECK-NEXT:    });
  // CHECK-NEXT:}
  testKernelPtr<<<dim3(1), dim3(1, 2)>>>(args.arg1, args.arg2, karg3int);

  // CHECK:   dpct::get_default_queue().submit(
  // CHECK-NEXT:     [&](cl::sycl::handler &cgh) {
  // CHECK-NEXT:      cgh.parallel_for<dpct_kernel_name<class testKernel_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:         cl::sycl::nd_range<3>(cl::sycl::range<3>(1, 2, 1) * cl::sycl::range<3>(3, 2, 1), cl::sycl::range<3>(3, 2, 1)),
  // CHECK-NEXT:         [=](cl::sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:           testKernel(karg1int, karg2int, karg3int, item_ct1);
  // CHECK-NEXT:         });
  // CHECK-NEXT:     });
  testKernel<<<dim3(1, 2), dim3(1, 2, 3)>>>(karg1int, karg2int, karg3int);

  // CHECK:   dpct::get_default_queue().submit(
  // CHECK-NEXT:     [&](cl::sycl::handler &cgh) {
  // CHECK-NEXT:       auto arr_karg3int_ct2 = arr[karg3int];
  // CHECK-EMPTY:
  // CHECK-NEXT:       cgh.parallel_for<dpct_kernel_name<class testKernel_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:         cl::sycl::nd_range<3>(cl::sycl::range<3>(1, 1, griddim[0]) * cl::sycl::range<3>(1, 1, griddim[1] + 2), cl::sycl::range<3>(1, 1, griddim[1] + 2)),
  // CHECK-NEXT:         [=](cl::sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:           testKernel(karg1int, karg2int, arr_karg3int_ct2, item_ct1);
  // CHECK-NEXT:         });
  // CHECK-NEXT:     });
  testKernel <<<griddim.x, griddim.y + 2 >>>(karg1int, karg2int, arr[karg3int]);

  // CHECK:   dpct::get_default_queue().submit(
  // CHECK-NEXT:     [&](cl::sycl::handler &cgh) {
  // CHECK-NEXT:       cgh.parallel_for<dpct_kernel_name<class helloFromGPU_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:         cl::sycl::nd_range<3>(cl::sycl::range<3>(1, 1, 2) * cl::sycl::range<3>(1, 1, 4), cl::sycl::range<3>(1, 1, 4)),
  // CHECK-NEXT:         [=](cl::sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:           helloFromGPU(23, item_ct1);
  // CHECK-NEXT:         });
  // CHECK-NEXT:     });
  helloFromGPU <<<2, 4>>>(23);

  // CHECK:   dpct::get_default_queue().submit(
  // CHECK-NEXT:     [&](cl::sycl::handler &cgh) {
  // CHECK-NEXT:       cgh.parallel_for<dpct_kernel_name<class helloFromGPU_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:         cl::sycl::nd_range<3>(cl::sycl::range<3>(1, 1, 2) * cl::sycl::range<3>(1, 1, 4), cl::sycl::range<3>(1, 1, 4)),
  // CHECK-NEXT:         [=](cl::sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:           helloFromGPU(item_ct1);
  // CHECK-NEXT:         });
  // CHECK-NEXT:     });
  helloFromGPU <<<2, 4>>>();

  // CHECK:   dpct::get_default_queue().submit(
  // CHECK-NEXT:     [&](cl::sycl::handler &cgh) {
  // CHECK-NEXT:       cgh.parallel_for<dpct_kernel_name<class helloFromGPU2_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:         cl::sycl::nd_range<3>(cl::sycl::range<3>(1, 1, 2) * cl::sycl::range<3>(1, 1, 3), cl::sycl::range<3>(1, 1, 3)),
  // CHECK-NEXT:         [=](cl::sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:           helloFromGPU2(item_ct1);
  // CHECK-NEXT:         });
  // CHECK-NEXT:     });
  helloFromGPU2 <<<2, 3>>>();

  // CHECK:   dpct::get_default_queue().submit(
  // CHECK-NEXT:     [&](cl::sycl::handler &cgh) {
  // CHECK-NEXT:       cgh.parallel_for<dpct_kernel_name<class helloFromGPU_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:         cl::sycl::nd_range<3>(cl::sycl::range<3>(1, 1, 2) * cl::sycl::range<3>(threaddim.get(2), threaddim.get(1), threaddim.get(0)), cl::sycl::range<3>(threaddim.get(2), threaddim.get(1), threaddim.get(0))),
  // CHECK-NEXT:         [=](cl::sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:           helloFromGPU(item_ct1);
  // CHECK-NEXT:         });
  // CHECK-NEXT:     });
  helloFromGPU<<<2, threaddim>>>();

  // CHECK:   dpct::get_default_queue().submit(
  // CHECK-NEXT:     [&](cl::sycl::handler &cgh) {
  // CHECK-NEXT:       cgh.parallel_for<dpct_kernel_name<class helloFromGPU_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:         cl::sycl::nd_range<3>(cl::sycl::range<3>(griddim.get(2), griddim.get(1), griddim.get(0)) * cl::sycl::range<3>(1, 1, 4), cl::sycl::range<3>(1, 1, 4)),
  // CHECK-NEXT:         [=](cl::sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:           helloFromGPU(item_ct1);
  // CHECK-NEXT:         });
  // CHECK-NEXT:     });
  helloFromGPU<<<griddim, 4>>>();
}

struct config {
  int b;
  struct subconfig {
    int d;
  } c;
};

// CHECK: void foo_kernel(int a, int b, int c) {}
__global__ void foo_kernel(int a, int b, int c) {}

class foo_class {
public:
  foo_class(int n) : a(n) {}

  // CHECK:  int run_foo() {
  // CHECK-NEXT:    dpct::get_default_queue().submit(
  // CHECK-NEXT:      [&](cl::sycl::handler &cgh) {
  // CHECK-NEXT:         auto a_ct0 = a;
  // CHECK-NEXT:         auto aa_b_ct1 = aa.b;
  // CHECK-NEXT:         auto aa_c_d_ct2 = aa.c.d;
  // CHECK-EMPTY:
  // CHECK-NEXT:        cgh.parallel_for<dpct_kernel_name<class foo_kernel_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:          cl::sycl::nd_range<3>(cl::sycl::range<3>(1, 1, 1), cl::sycl::range<3>(1, 1, 1)),
  // CHECK-NEXT:          [=](cl::sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:            foo_kernel(a_ct0, aa_b_ct1, aa_c_d_ct2);
  // CHECK-NEXT:          });
  // CHECK-NEXT:      });
  // CHECK-NEXT:  }
  int run_foo() {
    foo_kernel<<<1, 1>>>(a, aa.b, aa.c.d);
  }

private:
  int a;
  struct config aa;
};

__global__ void foo_kernel3(int *d) {
}
//CHECK:void run_foo(cl::sycl::range<3> c, cl::sycl::range<3> d) {
//CHECK-NEXT:  if (1)
//CHECK-NEXT:    {
//CHECK-NEXT:      std::pair<dpct::buffer_t, size_t> buf_ct0 = dpct::get_buffer_and_offset(0);
//CHECK-NEXT:      size_t offset_ct0 = buf_ct0.second;
//CHECK-NEXT:      dpct::get_default_queue().submit(
//CHECK-NEXT:        [&](cl::sycl::handler &cgh) {
//CHECK-NEXT:          auto acc_ct0 = buf_ct0.first.get_access<cl::sycl::access::mode::read_write>(cgh);
//CHECK-EMPTY:
//CHECK-NEXT:          cgh.parallel_for<dpct_kernel_name<class foo_kernel3_{{[a-f0-9]+}}>>(
//CHECK-NEXT:            cl::sycl::nd_range<3>(cl::sycl::range<3>(c.get(2), c.get(1), c.get(0)), cl::sycl::range<3>(1, 1, 1)),
//CHECK-NEXT:            [=](cl::sycl::nd_item<3> item_ct1) {
//CHECK-NEXT:              int *ct0 = (int *)(&acc_ct0[0] + offset_ct0);
//CHECK-NEXT:              foo_kernel3(ct0);
//CHECK-NEXT:            });
//CHECK-NEXT:        });
//CHECK-NEXT:    }
//CHECK-NEXT:}
void run_foo(dim3 c, dim3 d) {
  if (1)
    foo_kernel3<<<c, 1>>>(0);
}
//CHECK:void run_foo2(cl::sycl::range<3> c, cl::sycl::range<3> d) {
//CHECK-NEXT:  if (1)
//CHECK-NEXT:    {
//CHECK-NEXT:      std::pair<dpct::buffer_t, size_t> buf_ct0 = dpct::get_buffer_and_offset(0);
//CHECK-NEXT:      size_t offset_ct0 = buf_ct0.second;
//CHECK-NEXT:      dpct::get_default_queue().submit(
//CHECK-NEXT:        [&](cl::sycl::handler &cgh) {
//CHECK-NEXT:          auto acc_ct0 = buf_ct0.first.get_access<cl::sycl::access::mode::read_write>(cgh);
//CHECK-EMPTY:
//CHECK-NEXT:          auto dpct_global_range = c * d;
//CHECK-EMPTY:
//CHECK-NEXT:          cgh.parallel_for<dpct_kernel_name<class foo_kernel3_{{[a-f0-9]+}}>>(
//CHECK-NEXT:            cl::sycl::nd_range<3>(cl::sycl::range<3>(dpct_global_range.get(2), dpct_global_range.get(1), dpct_global_range.get(0)), cl::sycl::range<3>(d.get(2), d.get(1), d.get(0))),
//CHECK-NEXT:            [=](cl::sycl::nd_item<3> item_ct1) {
//CHECK-NEXT:              int *ct0 = (int *)(&acc_ct0[0] + offset_ct0);
//CHECK-NEXT:              foo_kernel3(ct0);
//CHECK-NEXT:            });
//CHECK-NEXT:        });
//CHECK-NEXT:    }
//CHECK-NEXT:  else
//CHECK-NEXT:    {
//CHECK-NEXT:      std::pair<dpct::buffer_t, size_t> buf_ct0 = dpct::get_buffer_and_offset(0);
//CHECK-NEXT:      size_t offset_ct0 = buf_ct0.second;
//CHECK-NEXT:      dpct::get_default_queue().submit(
//CHECK-NEXT:        [&](cl::sycl::handler &cgh) {
//CHECK-NEXT:          auto acc_ct0 = buf_ct0.first.get_access<cl::sycl::access::mode::read_write>(cgh);
//CHECK-EMPTY:
//CHECK-NEXT:          cgh.parallel_for<dpct_kernel_name<class foo_kernel3_{{[a-f0-9]+}}>>(
//CHECK-NEXT:            cl::sycl::nd_range<3>(cl::sycl::range<3>(c.get(2), c.get(1), c.get(0)), cl::sycl::range<3>(1, 1, 1)),
//CHECK-NEXT:            [=](cl::sycl::nd_item<3> item_ct1) {
//CHECK-NEXT:              int *ct0 = (int *)(&acc_ct0[0] + offset_ct0);
//CHECK-NEXT:              foo_kernel3(ct0);
//CHECK-NEXT:            });
//CHECK-NEXT:        });
//CHECK-NEXT:    }
//CHECK-NEXT:}
void run_foo2(dim3 c, dim3 d) {
  if (1)
    foo_kernel3<<<c, d>>>(0);
  else
    foo_kernel3<<<c, 1>>>(0);
}
//CHECK:void run_foo3(cl::sycl::range<3> c, cl::sycl::range<3> d) {
//CHECK-NEXT:  for (;;)
//CHECK-NEXT:    {
//CHECK-NEXT:      std::pair<dpct::buffer_t, size_t> buf_ct0 = dpct::get_buffer_and_offset(0);
//CHECK-NEXT:      size_t offset_ct0 = buf_ct0.second;
//CHECK-NEXT:      dpct::get_default_queue().submit(
//CHECK-NEXT:        [&](cl::sycl::handler &cgh) {
//CHECK-NEXT:          auto acc_ct0 = buf_ct0.first.get_access<cl::sycl::access::mode::read_write>(cgh);
//CHECK-EMPTY:
//CHECK-NEXT:          auto dpct_global_range = c * d;
//CHECK-EMPTY:
//CHECK-NEXT:          cgh.parallel_for<dpct_kernel_name<class foo_kernel3_{{[a-f0-9]+}}>>(
//CHECK-NEXT:            cl::sycl::nd_range<3>(cl::sycl::range<3>(dpct_global_range.get(2), dpct_global_range.get(1), dpct_global_range.get(0)), cl::sycl::range<3>(d.get(2), d.get(1), d.get(0))),
//CHECK-NEXT:            [=](cl::sycl::nd_item<3> item_ct1) {
//CHECK-NEXT:              int *ct0 = (int *)(&acc_ct0[0] + offset_ct0);
//CHECK-NEXT:              foo_kernel3(ct0);
//CHECK-NEXT:            });
//CHECK-NEXT:        });
//CHECK-NEXT:    }
//CHECK-NEXT:}
void run_foo3(dim3 c, dim3 d) {
  for (;;)
    foo_kernel3<<<c, d>>>(0);
}
//CHECK:void run_foo4(cl::sycl::range<3> c, cl::sycl::range<3> d) {
//CHECK-NEXT: while (1)
//CHECK-NEXT:   {
//CHECK-NEXT:     std::pair<dpct::buffer_t, size_t> buf_ct0 = dpct::get_buffer_and_offset(0);
//CHECK-NEXT:     size_t offset_ct0 = buf_ct0.second;
//CHECK-NEXT:     dpct::get_default_queue().submit(
//CHECK-NEXT:       [&](cl::sycl::handler &cgh) {
//CHECK-NEXT:          auto acc_ct0 = buf_ct0.first.get_access<cl::sycl::access::mode::read_write>(cgh);
//CHECK-EMPTY:
//CHECK-NEXT:         cgh.parallel_for<dpct_kernel_name<class foo_kernel3_{{[a-f0-9]+}}>>(
//CHECK-NEXT:           cl::sycl::nd_range<3>(cl::sycl::range<3>(c.get(2), c.get(1), c.get(0)), cl::sycl::range<3>(1, 1, 1)),
//CHECK-NEXT:           [=](cl::sycl::nd_item<3> item_ct1) {
//CHECK-NEXT:             int *ct0 = (int *)(&acc_ct0[0] + offset_ct0);
//CHECK-NEXT:             foo_kernel3(ct0);
//CHECK-NEXT:           });
//CHECK-NEXT:       });
//CHECK-NEXT:   }
//CHECK-NEXT:}
void run_foo4(dim3 c, dim3 d) {
 while (1)
   foo_kernel3<<<c, 1>>>(0);
}

//CHECK:dpct::shared_memory<float, 1> result(32);
//CHECK-NEXT:void my_kernel(float* result, cl::sycl::nd_item<3> item_ct1, float *resultInGroup) {
//CHECK-NEXT:  // __shared__ variable
//CHECK-NEXT:  resultInGroup[item_ct1.get_local_id(2)] = item_ct1.get_group(2);
//CHECK-NEXT:  memcpy(&result[item_ct1.get_group(2)*8], resultInGroup, sizeof(float)*8);
//CHECK-NEXT:}
//CHECK-NEXT:int run_foo5 () {
//CHECK-NEXT:  {
//CHECK-NEXT:    std::pair<dpct::buffer_t, size_t> result_buf_ct0 = dpct::get_buffer_and_offset(result.get_ptr());
//CHECK-NEXT:    size_t result_offset_ct0 = result_buf_ct0.second;
//CHECK-NEXT:    dpct::get_default_queue().submit(
//CHECK-NEXT:      [&](cl::sycl::handler &cgh) {
//CHECK-NEXT:        cl::sycl::accessor<float, 1, cl::sycl::access::mode::read_write, cl::sycl::access::target::local> resultInGroup_acc_ct1(cl::sycl::range<1>(8), cgh);
//CHECK-NEXT:        auto result_acc_ct0 = result_buf_ct0.first.get_access<cl::sycl::access::mode::read_write>(cgh);
//CHECK-EMPTY:
//CHECK-NEXT:        cgh.parallel_for<dpct_kernel_name<class my_kernel_{{[0-9a-z]+}}>>(
//CHECK-NEXT:          cl::sycl::nd_range<3>(cl::sycl::range<3>(1, 1, 4) * cl::sycl::range<3>(1, 1, 8), cl::sycl::range<3>(1, 1, 8)),
//CHECK-NEXT:          [=](cl::sycl::nd_item<3> item_ct1) {
//CHECK-NEXT:            float *result_ct0 = (float *)(&result_acc_ct0[0] + result_offset_ct0);
//CHECK-NEXT:            my_kernel(result_ct0, item_ct1, resultInGroup_acc_ct1.get_pointer());
//CHECK-NEXT:          });
//CHECK-NEXT:      });
//CHECK-NEXT:  }
//CHECK-NEXT:  printf("%f ", result[10]);
//CHECK-NEXT:}
 __managed__ float result[32];
__global__ void my_kernel(float* result) {
  __shared__ float resultInGroup[8]; // __shared__ variable
  resultInGroup[threadIdx.x] = blockIdx.x;
  memcpy(&result[blockIdx.x*8], resultInGroup, sizeof(float)*8);
}
int run_foo5 () {
  my_kernel<<<4, 8>>>(result);
  printf("%f ", result[10]);
}

//CHECK:dpct::shared_memory<float, 1> result2(32);
//CHECK-NEXT:int run_foo6 () {
//CHECK-NEXT:  {
//CHECK-NEXT:    std::pair<dpct::buffer_t, size_t> result2_buf_ct0 = dpct::get_buffer_and_offset(result2.get_ptr());
//CHECK-NEXT:    size_t result2_offset_ct0 = result2_buf_ct0.second;
//CHECK-NEXT:    dpct::get_default_queue().submit(
//CHECK-NEXT:      [&](cl::sycl::handler &cgh) {
//CHECK-NEXT:        cl::sycl::accessor<float, 1, cl::sycl::access::mode::read_write, cl::sycl::access::target::local> resultInGroup_acc_ct1(cl::sycl::range<1>(8), cgh);
//CHECK-NEXT:        auto result2_acc_ct0 = result2_buf_ct0.first.get_access<cl::sycl::access::mode::read_write>(cgh);
//CHECK-EMPTY:
//CHECK-NEXT:        cgh.parallel_for<dpct_kernel_name<class my_kernel_{{[0-9a-z]+}}>>(
//CHECK-NEXT:          cl::sycl::nd_range<3>(cl::sycl::range<3>(1, 1, 4) * cl::sycl::range<3>(1, 1, 8), cl::sycl::range<3>(1, 1, 8)),
//CHECK-NEXT:          [=](cl::sycl::nd_item<3> item_ct1) {
//CHECK-NEXT:            float *result2_ct0 = (float *)(&result2_acc_ct0[0] + result2_offset_ct0);
//CHECK-NEXT:            my_kernel(result2_ct0, item_ct1, resultInGroup_acc_ct1.get_pointer());
//CHECK-NEXT:          });
//CHECK-NEXT:      });
//CHECK-NEXT:  }
//CHECK-NEXT:  printf("%f ", result2[10]);
//CHECK-NEXT:}
 __managed__ float result2[32];
int run_foo6 () {
  my_kernel<<<4, 8>>>(result2);
  printf("%f ", result2[10]);
}

//CHECK:dpct::shared_memory<float, 0> result3;
//CHECK-NEXT:int run_foo7 () {
//CHECK-NEXT:  {
//CHECK-NEXT:    std::pair<dpct::buffer_t, size_t> result3_buf_ct0 = dpct::get_buffer_and_offset(result3.get_ptr());
//CHECK-NEXT:    size_t result3_offset_ct0 = result3_buf_ct0.second;
//CHECK-NEXT:    dpct::get_default_queue().submit(
//CHECK-NEXT:      [&](cl::sycl::handler &cgh) {
//CHECK-NEXT:        cl::sycl::accessor<float, 1, cl::sycl::access::mode::read_write, cl::sycl::access::target::local> resultInGroup_acc_ct1(cl::sycl::range<1>(8), cgh);
//CHECK-NEXT:        auto result3_acc_ct0 = result3_buf_ct0.first.get_access<cl::sycl::access::mode::read_write>(cgh);
//CHECK-EMPTY:
//CHECK-NEXT:        cgh.parallel_for<dpct_kernel_name<class my_kernel_{{[0-9a-z]+}}>>(
//CHECK-NEXT:          cl::sycl::nd_range<3>(cl::sycl::range<3>(1, 1, 4) * cl::sycl::range<3>(1, 1, 8), cl::sycl::range<3>(1, 1, 8)),
//CHECK-NEXT:          [=](cl::sycl::nd_item<3> item_ct1) {
//CHECK-NEXT:            float *result3_ct0 = (float *)(&result3_acc_ct0[0] + result3_offset_ct0);
//CHECK-NEXT:            my_kernel(result3_ct0, item_ct1, resultInGroup_acc_ct1.get_pointer());
//CHECK-NEXT:          });
//CHECK-NEXT:      });
//CHECK-NEXT:  }
//CHECK-NEXT:  printf("%f ", result3[0]);
//CHECK-NEXT:}
__managed__ float result3;
int run_foo7 () {
  my_kernel<<<4, 8>>>(&result3);
  printf("%f ", result3);
}
