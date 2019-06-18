// FIXME
// UNSUPPORTED: -windows-
// RUN: syclct -out-root %T %s -- -x cuda --cuda-host-only --cuda-path="%cuda-path"
// RUN: FileCheck --input-file %T/template-kernel-call.sycl.cpp --match-full-lines %s

void printf(const char *format, unsigned char data);

template <class TName, unsigned N, class TData>
// CHECK: void testKernelPtr(const TData *L, const TData *M, cl::sycl::nd_item<3> [[ITEMNAME:item_[a-f0-9]+]]) {
__global__ void testKernelPtr(const TData *L, const TData *M) {
  // CHECK: int gtid = [[ITEMNAME]].get_group(0) * [[ITEMNAME]].get_local_range().get(0) + [[ITEMNAME]].get_local_id(0);
  int gtid = blockIdx.x * blockDim.x + threadIdx.x;
}

template<class TData>
// CHECK: void testKernel(TData L, TData M, int N, cl::sycl::nd_item<3> [[ITEMNAME:item_[a-f0-9]+]]) {
__global__ void testKernel(TData L, TData M, int N) {
  // CHECK: int gtid = [[ITEMNAME]].get_group(0) * [[ITEMNAME]].get_local_range().get(0) + [[ITEMNAME]].get_local_id(0);
  int gtid = blockIdx.x * blockDim.x + threadIdx.x;
  L = M;
}

// CHECK: struct __sycl_align__(8) LA {
struct __align__(8) LA {
  unsigned int l, a;
};

template<class T>
class TestTemplate {
public:
  T data;
};

const unsigned ktarg = 80;
dim3 griddim = 2;
dim3 threaddim = 32;

template<class T>
void runTest() {
  typedef TestTemplate<T> TT;
  const void *karg1 = 0;
  const T *karg2 = 0;
  T *karg3 = 0;
  const TestTemplate<T> *karg4 = 0;
  TT *karg5 = 0;

  // CHECK:  {
  // CHECK-NEXT:    std::pair<syclct::buffer_t, size_t> karg1_buf = syclct::get_buffer_and_offset(karg1);
  // CHECK-NEXT:    size_t karg1_offset = karg1_buf.second;
  // CHECK-NEXT:    std::pair<syclct::buffer_t, size_t> karg2_buf = syclct::get_buffer_and_offset(karg2);
  // CHECK-NEXT:    size_t karg2_offset = karg2_buf.second;
  // CHECK-NEXT:    syclct::get_default_queue().submit(
  // CHECK-NEXT:      [&](cl::sycl::handler &cgh) {
  // CHECK-NEXT:        auto karg1_acc = karg1_buf.first.get_access<cl::sycl::access::mode::read_write>(cgh);
  // CHECK-NEXT:        auto karg2_acc = karg2_buf.first.get_access<cl::sycl::access::mode::read_write>(cgh);
  // CHECK-NEXT:        cgh.parallel_for<syclct_kernel_name<class testKernelPtr_{{[a-f0-9]+}}, class TestName, syclct_kernel_scalar<ktarg>, T>>(
  // CHECK-NEXT:          cl::sycl::nd_range<3>((griddim * threaddim), threaddim),
  // CHECK-NEXT:          [=](cl::sycl::nd_item<3> [[ITEM:item_[a-f0-9]+]]) {
  // CHECK-NEXT:            const void *karg1 = (const void*)(&karg1_acc[0] + karg1_offset);
  // CHECK-NEXT:            const T *karg2 = (const T*)(&karg2_acc[0] + karg2_offset);
  // CHECK-NEXT:            testKernelPtr<class TestName, ktarg, T>((const T *)karg1, karg2, [[ITEM]]);
  // CHECK-NEXT:          });
  // CHECK-NEXT:      });
  // CHECK-NEXT:  }
  testKernelPtr<class TestName, ktarg, T><<<griddim, threaddim>>>((const T *)karg1, karg2);

  // CHECK:  {
  // CHECK-NEXT:    std::pair<syclct::buffer_t, size_t> karg1_buf = syclct::get_buffer_and_offset(karg1);
  // CHECK-NEXT:    size_t karg1_offset = karg1_buf.second;
  // CHECK-NEXT:    std::pair<syclct::buffer_t, size_t> karg3_buf = syclct::get_buffer_and_offset(karg3);
  // CHECK-NEXT:    size_t karg3_offset = karg3_buf.second;
  // CHECK-NEXT:    syclct::get_default_queue().submit(
  // CHECK-NEXT:      [&](cl::sycl::handler &cgh) {
  // CHECK-NEXT:        auto karg1_acc = karg1_buf.first.get_access<cl::sycl::access::mode::read_write>(cgh);
  // CHECK-NEXT:        auto karg3_acc = karg3_buf.first.get_access<cl::sycl::access::mode::read_write>(cgh);
  // CHECK-NEXT:        cgh.parallel_for<syclct_kernel_name<class testKernelPtr_{{[a-f0-9]+}}, class TestTemplate<T>, syclct_kernel_scalar<ktarg>, T>>(
  // CHECK-NEXT:          cl::sycl::nd_range<3>((griddim * threaddim), threaddim),
  // CHECK-NEXT:          [=](cl::sycl::nd_item<3> [[ITEM:item_[a-f0-9]+]]) {
  // CHECK-NEXT:            const void *karg1 = (const void*)(&karg1_acc[0] + karg1_offset);
  // CHECK-NEXT:            T *karg3 = (T*)(&karg3_acc[0] + karg3_offset);
  // CHECK-NEXT:            testKernelPtr<class TestTemplate<T>, ktarg, T>(karg1, karg3, [[ITEM]]);
  // CHECK-NEXT:          });
  // CHECK-NEXT:      });
  // CHECK-NEXT:  }
  testKernelPtr<class TestTemplate<T>, ktarg, T><<<griddim, threaddim>>>(karg1, karg3);

  // CHECK:  {
  // CHECK-NEXT:    std::pair<syclct::buffer_t, size_t> karg4_buf = syclct::get_buffer_and_offset(karg4);
  // CHECK-NEXT:    size_t karg4_offset = karg4_buf.second;
  // CHECK-NEXT:    std::pair<syclct::buffer_t, size_t> karg5_buf = syclct::get_buffer_and_offset(karg5);
  // CHECK-NEXT:    size_t karg5_offset = karg5_buf.second;
  // CHECK-NEXT:    syclct::get_default_queue().submit(
  // CHECK-NEXT:      [&](cl::sycl::handler &cgh) {
  // CHECK-NEXT:        auto karg4_acc = karg4_buf.first.get_access<cl::sycl::access::mode::read_write>(cgh);
  // CHECK-NEXT:        auto karg5_acc = karg5_buf.first.get_access<cl::sycl::access::mode::read_write>(cgh);
  // CHECK-NEXT:        cgh.parallel_for<syclct_kernel_name<class testKernelPtr_{{[a-f0-9]+}}, T, syclct_kernel_scalar<ktarg>, TestTemplate<T>>>(
  // CHECK-NEXT:          cl::sycl::nd_range<3>((griddim * threaddim), threaddim),
  // CHECK-NEXT:          [=](cl::sycl::nd_item<3> [[ITEM:item_[a-f0-9]+]]) {
  // CHECK-NEXT:            const TestTemplate<T> *karg4 = (const TestTemplate<T>*)(&karg4_acc[0] + karg4_offset);
  // CHECK-NEXT:            TT *karg5 = (TT*)(&karg5_acc[0] + karg5_offset);
  // CHECK-NEXT:            testKernelPtr<T, ktarg, TestTemplate<T>>(karg4, karg5, [[ITEM]]);
  // CHECK-NEXT:          });
  // CHECK-NEXT:      });
  // CHECK-NEXT:  }
  testKernelPtr<T, ktarg, TestTemplate<T> ><<<griddim, threaddim>>>(karg4, karg5);

  T karg1T, karg2T;
  // CHECK:  {
  // CHECK-NEXT:    syclct::get_default_queue().submit(
  // CHECK-NEXT:      [&](cl::sycl::handler &cgh) {
  // CHECK-NEXT:        cgh.parallel_for<syclct_kernel_name<class testKernel_{{[a-f0-9]+}}, T>>(
  // CHECK-NEXT:          cl::sycl::nd_range<3>((griddim * threaddim), threaddim),
  // CHECK-NEXT:          [=](cl::sycl::nd_item<3> [[ITEM:item_[a-f0-9]+]]) {
  // CHECK-NEXT:            testKernel<T>(karg1T, karg2T, ktarg, [[ITEM]]);
  // CHECK-NEXT:          });
  // CHECK-NEXT:      });
  // CHECK-NEXT:  }
  testKernel<T><<<griddim, threaddim>>>(karg1T, karg2T, ktarg);

  TestTemplate<T> karg3TT;
  TT karg4TT;

  // CHECK:  {
  // CHECK-NEXT:    syclct::get_default_queue().submit(
  // CHECK-NEXT:      [&](cl::sycl::handler &cgh) {
  // CHECK-NEXT:        cgh.parallel_for<syclct_kernel_name<class testKernel_{{[a-f0-9]+}}, TestTemplate<T>>>(
  // CHECK-NEXT:          cl::sycl::nd_range<3>((griddim * threaddim), threaddim),
  // CHECK-NEXT:          [=](cl::sycl::nd_item<3> [[ITEM:item_[a-f0-9]+]]) {
  // CHECK-NEXT:            testKernel<TestTemplate<T>>(karg3TT, karg4TT, ktarg, [[ITEM]]);
  // CHECK-NEXT:          });
  // CHECK-NEXT:      });
  // CHECK-NEXT:  }
  testKernel<TestTemplate<T> ><<<griddim, threaddim>>>(karg3TT, karg4TT, ktarg);

  // CHECK:  {
  // CHECK-NEXT:    syclct::get_default_queue().submit(
  // CHECK-NEXT:      [&](cl::sycl::handler &cgh) {
  // CHECK-NEXT:        cgh.parallel_for<syclct_kernel_name<class testKernel_{{[a-f0-9]+}}, TT>>(
  // CHECK-NEXT:          cl::sycl::nd_range<3>((griddim * threaddim), threaddim),
  // CHECK-NEXT:          [=](cl::sycl::nd_item<3> [[ITEM:item_[a-f0-9]+]]) {
  // CHECK-NEXT:            testKernel<TT>(karg3TT, karg4TT, ktarg, [[ITEM]]);
  // CHECK-NEXT:          });
  // CHECK-NEXT:      });
  // CHECK-NEXT:  }
  testKernel<TT><<<griddim, threaddim>>>(karg3TT, karg4TT, ktarg);
}

int main() {
  void *karg1 = 0;
  LA *karg2 = 0;
  // CHECK:  {
  // CHECK-NEXT:    std::pair<syclct::buffer_t, size_t> karg1_buf = syclct::get_buffer_and_offset(karg1);
  // CHECK-NEXT:    size_t karg1_offset = karg1_buf.second;
  // CHECK-NEXT:    std::pair<syclct::buffer_t, size_t> karg2_buf = syclct::get_buffer_and_offset(karg2);
  // CHECK-NEXT:    size_t karg2_offset = karg2_buf.second;
  // CHECK-NEXT:    syclct::get_default_queue().submit(
  // CHECK-NEXT:      [&](cl::sycl::handler &cgh) {
  // CHECK-NEXT:        auto karg1_acc = karg1_buf.first.get_access<cl::sycl::access::mode::read_write>(cgh);
  // CHECK-NEXT:        auto karg2_acc = karg2_buf.first.get_access<cl::sycl::access::mode::read_write>(cgh);
  // CHECK-NEXT:        cgh.parallel_for<syclct_kernel_name<class testKernelPtr_{{[a-f0-9]+}}, class TestName, syclct_kernel_scalar<ktarg>, LA>>(
  // CHECK-NEXT:          cl::sycl::nd_range<3>((griddim * threaddim), threaddim),
  // CHECK-NEXT:          [=](cl::sycl::nd_item<3> [[ITEM:item_[a-f0-9]+]]) {
  // CHECK-NEXT:            void *karg1 = (void*)(&karg1_acc[0] + karg1_offset);
  // CHECK-NEXT:            LA *karg2 = (LA*)(&karg2_acc[0] + karg2_offset);
  // CHECK-NEXT:            testKernelPtr<class TestName, ktarg, LA>((const LA *)karg1, karg2, [[ITEM]]);
  // CHECK-NEXT:          });
  // CHECK-NEXT:      });
  // CHECK-NEXT:  }
  testKernelPtr<class TestName, ktarg, LA><<<griddim, threaddim>>>((const LA *)karg1, karg2);

  LA karg1LA, karg2LA;
  int intvar = 20;
  // CHECK:  {
  // CHECK-NEXT:    syclct::get_default_queue().submit(
  // CHECK-NEXT:      [&](cl::sycl::handler &cgh) {
  // CHECK-NEXT:        cgh.parallel_for<syclct_kernel_name<class testKernel_{{[a-f0-9]+}}, LA>>(
  // CHECK-NEXT:          cl::sycl::nd_range<3>((cl::sycl::range<3>(10, 1, 1) * cl::sycl::range<3>(intvar, 1, 1)), cl::sycl::range<3>(intvar, 1, 1)),
  // CHECK-NEXT:          [=](cl::sycl::nd_item<3> [[ITEM:item_[a-f0-9]+]]) {
  // CHECK-NEXT:            testKernel<LA>(karg1LA, karg2LA, ktarg, [[ITEM]]);
  // CHECK-NEXT:          });
  // CHECK-NEXT:      });
  // CHECK-NEXT:  }
  testKernel<LA><<<10, intvar>>>(karg1LA, karg2LA, ktarg);
}
