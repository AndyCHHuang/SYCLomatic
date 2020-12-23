// RUN: dpct --format-range=none -out-root %T/cufft-deduce %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only -std=c++14
// RUN: FileCheck --input-file %T/cufft-deduce/cufft-deduce.dp.cpp --match-full-lines %s
#include <cstdio>
#include <cufft.h>
#include <cuda_runtime.h>


size_t* work_size;
int odist;
int ostride;
int * onembed;
int idist;
int istride;
int* inembed;
int * n;
constexpr int rank = 3;

//CHECK:void foo1(std::shared_ptr<oneapi::mkl::dft::descriptor<oneapi::mkl::dft::precision::DOUBLE, oneapi::mkl::dft::domain::REAL>> plan) {
//CHECK-NEXT:  double* odata;
//CHECK-NEXT:  sycl::double2* idata;
//CHECK-NEXT:  if ((void *)idata == (void *)odata) {
//CHECK-NEXT:  oneapi::mkl::dft::compute_backward(*plan, (double*)idata);
//CHECK-NEXT:  } else {
//CHECK-NEXT:  oneapi::mkl::dft::compute_backward(*plan, (double*)idata, odata);
//CHECK-NEXT:  }
//CHECK-NEXT:}
void foo1(cufftHandle plan) {
  double* odata;
  double2* idata;
  cufftExecZ2D(plan, idata, odata);
}

//CHECK:void foo2(std::shared_ptr<oneapi::mkl::dft::descriptor<oneapi::mkl::dft::precision::DOUBLE, oneapi::mkl::dft::domain::REAL>> plan) {
//CHECK-NEXT:  double* odata;
//CHECK-NEXT:  sycl::double2* idata;
//CHECK-NEXT:  if ((void *)idata == (void *)odata) {
//CHECK-NEXT:  oneapi::mkl::dft::compute_backward(*plan, (double*)idata);
//CHECK-NEXT:  } else {
//CHECK-NEXT:  oneapi::mkl::dft::compute_backward(*plan, (double*)idata, odata);
//CHECK-NEXT:  }
//CHECK-NEXT:}
void foo2(cufftHandle plan) {
  double* odata;
  double2* idata;
  cufftExecZ2D(plan, idata, odata);
}

int main() {
  //CHECK:constexpr int type = 108;
  constexpr cufftType_t type = CUFFT_Z2D;
  cufftType_t type2 = type;

  //CHECK:std::shared_ptr<oneapi::mkl::dft::descriptor<oneapi::mkl::dft::precision::DOUBLE, oneapi::mkl::dft::domain::REAL>> plan1;
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1067:{{[0-9]+}}: The argument work_size is not supported in the migrated API. You may need to adjust the code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1071:{{[0-9]+}}: The placement of the FFT computational function cannot be deduced. It is migrated as out-of-place. You may need to adjust the code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:plan1 = std::make_shared<oneapi::mkl::dft::descriptor<oneapi::mkl::dft::precision::DOUBLE, oneapi::mkl::dft::domain::REAL>>(std::vector<std::int64_t>{n[0], n[1], n[2]});
  //CHECK-NEXT:plan1->set_value(oneapi::mkl::dft::config_param::PLACEMENT, DFTI_CONFIG_VALUE::DFTI_NOT_INPLACE);
  //CHECK-NEXT:plan1->set_value(oneapi::mkl::dft::config_param::NUMBER_OF_TRANSFORMS, 12);
  //CHECK-NEXT:if (inembed != nullptr && onembed != nullptr) {
  //CHECK-NEXT:std::int64_t input_stride_ct{{[0-9]+}}[4] = {0, inembed[2] * inembed[1] * istride, inembed[2] * istride, istride};
  //CHECK-NEXT:std::int64_t output_stride_ct{{[0-9]+}}[4] = {0, onembed[2] * onembed[1] * ostride, onembed[2] * ostride, ostride};
  //CHECK-NEXT:plan1->set_value(oneapi::mkl::dft::config_param::FWD_DISTANCE, odist);
  //CHECK-NEXT:plan1->set_value(oneapi::mkl::dft::config_param::BWD_DISTANCE, idist);
  //CHECK-NEXT:plan1->set_value(oneapi::mkl::dft::config_param::INPUT_STRIDES, input_stride_ct{{[0-9]+}});
  //CHECK-NEXT:plan1->set_value(oneapi::mkl::dft::config_param::OUTPUT_STRIDES, output_stride_ct{{[0-9]+}});
  //CHECK-NEXT:} else {
  //CHECK-NEXT:std::int64_t input_stride_ct{{[0-9]+}}[4] = {0, n[1]*(n[2]/2+1), (n[2]/2+1), 1};
  //CHECK-NEXT:plan1->set_value(oneapi::mkl::dft::config_param::INPUT_STRIDES, input_stride_ct{{[0-9]+}});
  //CHECK-NEXT:plan1->set_value(oneapi::mkl::dft::config_param::FWD_DISTANCE, n[2]*n[1]*n[0]);
  //CHECK-NEXT:plan1->set_value(oneapi::mkl::dft::config_param::BWD_DISTANCE, n[2]*n[1]*(n[0]/2+1));
  //CHECK-NEXT:}
  //CHECK-NEXT:plan1->commit(q_ct1);
  cufftHandle plan1;
  cufftMakePlanMany(plan1, rank, n, inembed, istride, idist, onembed, ostride, odist, type, 12, work_size);

  //CHECK:std::shared_ptr<oneapi::mkl::dft::descriptor<oneapi::mkl::dft::precision::DOUBLE, oneapi::mkl::dft::domain::REAL>> plan2;
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1067:{{[0-9]+}}: The argument work_size is not supported in the migrated API. You may need to adjust the code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1071:{{[0-9]+}}: The placement of the FFT computational function cannot be deduced. It is migrated as out-of-place. You may need to adjust the code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:plan2 = std::make_shared<oneapi::mkl::dft::descriptor<oneapi::mkl::dft::precision::DOUBLE, oneapi::mkl::dft::domain::REAL>>(std::vector<std::int64_t>{n[0], n[1], n[2]});
  //CHECK-NEXT:plan2->set_value(oneapi::mkl::dft::config_param::PLACEMENT, DFTI_CONFIG_VALUE::DFTI_NOT_INPLACE);
  //CHECK-NEXT:plan2->set_value(oneapi::mkl::dft::config_param::NUMBER_OF_TRANSFORMS, 12);
  //CHECK-NEXT:if (inembed != nullptr && onembed != nullptr) {
  //CHECK-NEXT:std::int64_t input_stride_ct{{[0-9]+}}[4] = {0, inembed[2] * inembed[1] * istride, inembed[2] * istride, istride};
  //CHECK-NEXT:std::int64_t output_stride_ct{{[0-9]+}}[4] = {0, onembed[2] * onembed[1] * ostride, onembed[2] * ostride, ostride};
  //CHECK-NEXT:plan2->set_value(oneapi::mkl::dft::config_param::FWD_DISTANCE, odist);
  //CHECK-NEXT:plan2->set_value(oneapi::mkl::dft::config_param::BWD_DISTANCE, idist);
  //CHECK-NEXT:plan2->set_value(oneapi::mkl::dft::config_param::INPUT_STRIDES, input_stride_ct{{[0-9]+}});
  //CHECK-NEXT:plan2->set_value(oneapi::mkl::dft::config_param::OUTPUT_STRIDES, output_stride_ct{{[0-9]+}});
  //CHECK-NEXT:} else {
  //CHECK-NEXT:std::int64_t input_stride_ct{{[0-9]+}}[4] = {0, n[1]*(n[2]/2+1), (n[2]/2+1), 1};
  //CHECK-NEXT:plan2->set_value(oneapi::mkl::dft::config_param::INPUT_STRIDES, input_stride_ct{{[0-9]+}});
  //CHECK-NEXT:plan2->set_value(oneapi::mkl::dft::config_param::FWD_DISTANCE, n[2]*n[1]*n[0]);
  //CHECK-NEXT:plan2->set_value(oneapi::mkl::dft::config_param::BWD_DISTANCE, n[2]*n[1]*(n[0]/2+1));
  //CHECK-NEXT:}
  //CHECK-NEXT:plan2->commit(q_ct1);
  cufftHandle plan2;
  cufftMakePlanMany(plan2, rank, n, inembed, istride, idist, onembed, ostride, odist, type, 12, work_size);


  //CHECK:std::shared_ptr<oneapi::mkl::dft::descriptor<oneapi::mkl::dft::precision::DOUBLE, oneapi::mkl::dft::domain::REAL>> plan3;
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1067:{{[0-9]+}}: The argument work_size is not supported in the migrated API. You may need to adjust the code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:/*
  //CHECK-NEXT:DPCT1071:{{[0-9]+}}: The placement of the FFT computational function cannot be deduced. It is migrated as out-of-place. You may need to adjust the code.
  //CHECK-NEXT:*/
  //CHECK-NEXT:plan3 = std::make_shared<oneapi::mkl::dft::descriptor<oneapi::mkl::dft::precision::DOUBLE, oneapi::mkl::dft::domain::REAL>>(std::vector<std::int64_t>{n[0], n[1], n[2]});
  //CHECK-NEXT:plan3->set_value(oneapi::mkl::dft::config_param::PLACEMENT, DFTI_CONFIG_VALUE::DFTI_NOT_INPLACE);
  //CHECK-NEXT:plan3->set_value(oneapi::mkl::dft::config_param::NUMBER_OF_TRANSFORMS, 12);
  //CHECK-NEXT:if (inembed != nullptr && onembed != nullptr) {
  //CHECK-NEXT:std::int64_t input_stride_ct{{[0-9]+}}[4] = {0, inembed[2] * inembed[1] * istride, inembed[2] * istride, istride};
  //CHECK-NEXT:std::int64_t output_stride_ct{{[0-9]+}}[4] = {0, onembed[2] * onembed[1] * ostride, onembed[2] * ostride, ostride};
  //CHECK-NEXT:plan3->set_value(oneapi::mkl::dft::config_param::FWD_DISTANCE, odist);
  //CHECK-NEXT:plan3->set_value(oneapi::mkl::dft::config_param::BWD_DISTANCE, idist);
  //CHECK-NEXT:plan3->set_value(oneapi::mkl::dft::config_param::INPUT_STRIDES, input_stride_ct{{[0-9]+}});
  //CHECK-NEXT:plan3->set_value(oneapi::mkl::dft::config_param::OUTPUT_STRIDES, output_stride_ct{{[0-9]+}});
  //CHECK-NEXT:} else {
  //CHECK-NEXT:std::int64_t input_stride_ct{{[0-9]+}}[4] = {0, n[1]*(n[2]/2+1), (n[2]/2+1), 1};
  //CHECK-NEXT:plan3->set_value(oneapi::mkl::dft::config_param::INPUT_STRIDES, input_stride_ct{{[0-9]+}});
  //CHECK-NEXT:plan3->set_value(oneapi::mkl::dft::config_param::FWD_DISTANCE, n[2]*n[1]*n[0]);
  //CHECK-NEXT:plan3->set_value(oneapi::mkl::dft::config_param::BWD_DISTANCE, n[2]*n[1]*(n[0]/2+1));
  //CHECK-NEXT:}
  //CHECK-NEXT:plan3->commit(q_ct1);
  cufftHandle plan3;
  cufftMakePlanMany(plan3, rank, n, inembed, istride, idist, onembed, ostride, odist, type2, 12, work_size);

  foo1(plan1);
  foo2(plan2);

  return 0;
}

