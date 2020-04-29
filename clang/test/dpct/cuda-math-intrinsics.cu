// RUN: dpct --format-range=none -out-root %T %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/cuda-math-intrinsics.dp.cpp --match-full-lines %s

#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <algorithm>

#include <stdio.h>

// CHECK: #include <algorithm>

#include "cuda_fp16.h"

using namespace std;

// CHECK: dpct::constant_memory<double, 0> d;
// CHECK-NEXT: dpct::constant_memory<double, 0> d2;
__constant__ double d;
__constant__ double d2;

// CHECK: double test(double d3, double d) {
// CHECK-NEXT:  return sycl::max(d, d3);
// CHECK-NEXT:}
__device__ double test(double d3) {
  return max(d, d3);
}

// CHECK:  double test2(double d, double d2) {
// CHECK-NEXT:   return sycl::max(d, d2);
// CHECK-NEXT: }
__device__ double test2() {
  return max(d, d2);
}

// CHECK:  double test3(double d4, double d5) {
// CHECK-NEXT:   return sycl::max(d4, d5);
// CHECK-NEXT: }
__device__ double test3(double d4, double d5) {
  return max(d4, d5);
}

// CHECK: dpct::constant_memory<float, 0> C;
// CHECK-NEXT:  int foo(int n, float C) {
// CHECK-NEXT:   return n == 1 ? C : 0;
// CHECK-NEXT: }
__constant__ float C;
__device__ int foo(int n) {
  return n == 1 ? C : 0;
}

__global__ void kernelFuncHalf(double *deviceArrayDouble) {
  __half h, h_1, h_2;
  __half2 h2, h2_1, h2_2;
  bool b;

  // Half Arithmetic Functions

  // TODO:1CHECK: h2_2 = h2 / h2_1;
  //h2_2 = __h2div(h2, h2_1);
  // TODO:1CHECK: h_2 = h / h_1;
  //h_2 = __hdiv(h, h_1);
  // CHECK: h_2 = sycl::fma(h, h_1, h_2);
  h_2 = __hfma(h, h_1, h_2);
  // CHECK: h_2 = h * h_1;
  h_2 = __hmul(h, h_1);
  // CHECK: h_2 = -h;
  h_2 = __hneg(h);
  // CHECK: h_2 = h - h_1;
  h_2 = __hsub(h, h_1);

  // Half2 Arithmetic Functions

  // CHECK: h2_2 = sycl::fma(h2, h2_1, h2_2);
  h2_2 = __hfma2(h2, h2_1, h2_2);
  // CHECK: h2_2 = h2 * h2_1;
  h2_2 = __hmul2(h2, h2_1);
  // CHECK: h2_2 = -h2;
  h2_2 = __hneg2(h2);
  // CHECK: h2_2 = h2 - h2_1;
  h2_2 = __hsub2(h2, h2_1);

  // Half Comparison Functions

  // CHECK: b = h == h_1;
  b = __heq(h, h_1);
  // CHECK: b = h >= h_1;
  b = __hge(h, h_1);
  // CHECK: b = h > h_1;
  b = __hgt(h, h_1);
  // CHECK: b = sycl::isinf(h);
  b = __hisinf(h);
  // CHECK: b = sycl::isnan(h);
  b = __hisnan(h);
  // CHECK: b = h <= h_1;
  b = __hle(h, h_1);
  // CHECK: b = h < h_1;
  b = __hlt(h, h_1);
  // CHECK: b = h != h_1;
  b = __hne(h, h_1);

  // Half2 Comparison Functions

  // CHECK: /*
  // CHECK-NEXT: DPCT1004:{{[0-9]+}}: Could not generate replacement.
  // CHECK-NEXT: */
  h2_2 = __heq2(h2, h2_1);
  // CHECK: /*
  // CHECK-NEXT: DPCT1004:{{[0-9]+}}: Could not generate replacement.
  // CHECK-NEXT: */
  h2_2 = __hge2(h2, h2_1);
  // CHECK: /*
  // CHECK-NEXT: DPCT1004:{{[0-9]+}}: Could not generate replacement.
  // CHECK-NEXT: */
  h2_2 = __hgt2(h2, h2_1);
  // CHECK: /*
  // CHECK-NEXT: DPCT1004:{{[0-9]+}}: Could not generate replacement.
  // CHECK-NEXT: */
  h2_2 = __hisnan2(h2);
  // CHECK: /*
  // CHECK-NEXT: DPCT1004:{{[0-9]+}}: Could not generate replacement.
  // CHECK-NEXT: */
  h2_2 = __hle2(h2, h2_1);
  // CHECK: /*
  // CHECK-NEXT: DPCT1004:{{[0-9]+}}: Could not generate replacement.
  // CHECK-NEXT: */
  h2_2 = __hlt2(h2, h2_1);
  // CHECK: /*
  // CHECK-NEXT: DPCT1004:{{[0-9]+}}: Could not generate replacement.
  // CHECK-NEXT: */
  h2_2 = __hne2(h2, h2_1);

  // Half Math Functions

  // CHECK: h_2 = sycl::ceil(h);
  h_2 = hceil(h);
  // CHECK: h_2 = sycl::cos(h);
  h_2 = hcos(h);
  // CHECK: h_2 = sycl::exp(h);
  h_2 = hexp(h);
  // CHECK: h_2 = sycl::exp10(h);
  h_2 = hexp10(h);
  // CHECK: h_2 = sycl::exp2(h);
  h_2 = hexp2(h);
  // CHECK: h_2 = sycl::floor(h);
  h_2 = hfloor(h);
  // CHECK: h_2 = sycl::log(h);
  h_2 = hlog(h);
  // CHECK: h_2 = sycl::log10(h);
  h_2 = hlog10(h);
  // CHECK: h_2 = sycl::log2(h);
  h_2 = hlog2(h);
  // CHECK: /*
  // CHECK-NEXT: DPCT1004:{{[0-9]+}}: Could not generate replacement.
  // CHECK-NEXT: */
  h_2 = hrcp(h);
  // CHECK: h_2 = sycl::rint(h);
  h_2 = hrint(h);
  // CHECK: h_2 = sycl::rsqrt(h);
  h_2 = hrsqrt(h);
  // CHECK: h_2 = sycl::sin(h);
  h_2 = hsin(h);
  // CHECK: h_2 = sycl::sqrt(h);
  h_2 = hsqrt(h);
  // CHECK: h_2 = sycl::trunc(h);
  h_2 = htrunc(h);

  // Half2 Math Functions

  // CHECK: h2_2 = sycl::ceil(h2);
  h2_2 = h2ceil(h2);
  // CHECK: h2_2 = sycl::cos(h2);
  h2_2 = h2cos(h2);
  // CHECK: h2_2 = sycl::exp(h2);
  h2_2 = h2exp(h2);
  // CHECK: h2_2 = sycl::exp10(h2);
  h2_2 = h2exp10(h2);
  // CHECK: h2_2 = sycl::exp2(h2);
  h2_2 = h2exp2(h2);
  // CHECK: h2_2 = sycl::floor(h2);
  h2_2 = h2floor(h2);
  // CHECK: h2_2 = sycl::log(h2);
  h2_2 = h2log(h2);
  // CHECK: h2_2 = sycl::log10(h2);
  h2_2 = h2log10(h2);
  // CHECK: h2_2 = sycl::log2(h2);
  h2_2 = h2log2(h2);
  // CHECK: /*
  // CHECK-NEXT: DPCT1004:{{[0-9]+}}: Could not generate replacement.
  // CHECK-NEXT: */
  h2_2 = h2rcp(h2);
  // CHECK: h2_2 = sycl::rint(h2);
  h2_2 = h2rint(h2);
  // CHECK: h2_2 = sycl::rsqrt(h2);
  h2_2 = h2rsqrt(h2);
  // CHECK: h2_2 = sycl::sin(h2);
  h2_2 = h2sin(h2);
  // CHECK: h2_2 = sycl::sqrt(h2);
  h2_2 = h2sqrt(h2);
  // CHECK: h2_2 = sycl::trunc(h2);
  h2_2 = h2trunc(h2);
}

__global__ void kernelFuncDouble(double *deviceArrayDouble) {
  double &d0 = *deviceArrayDouble, &d1 = *(deviceArrayDouble + 1), &d2 = *(deviceArrayDouble + 2);
  int i;

  // Double Precision Mathematical Functions

  // CHECK: d2 = sycl::acos(d0);
  d2 = acos(d0);
  // CHECK: d2 = sycl::acos((double)i);
  d2 = acos(i);

  // CHECK: d2 = sycl::acosh(d0);
  d2 = acosh(d0);
  // CHECK: d2 = sycl::acosh((double)i);
  d2 = acosh(i);

  // CHECK: d2 = sycl::asin(d0);
  d2 = asin(d0);
  // CHECK: d2 = sycl::asin((double)i);
  d2 = asin(i);

  // CHECK: d2 = sycl::asinh(d0);
  d2 = asinh(d0);
  // CHECK: d2 = sycl::asinh((double)i);
  d2 = asinh(i);

  // CHECK: d2 = sycl::atan2(d0, d1);
  d2 = atan2(d0, d1);
  // CHECK: d2 = sycl::atan2((double)i, (double)i);
  d2 = atan2(i, i);
  // CHECK: d2 = sycl::atan2(d0, (double)i);
  d2 = atan2(d0, i);
  // CHECK: d2 = sycl::atan2((double)i, d1);
  d2 = atan2(i, d1);

  // CHECK: d2 = sycl::atan(d0);
  d2 = atan(d0);
  // CHECK: d2 = sycl::atan((double)i);
  d2 = atan(i);

  // CHECK: d2 = sycl::atanh(d0);
  d2 = atanh(d0);
  // CHECK: d2 = sycl::atanh((double)i);
  d2 = atanh(i);

  // CHECK: d2 = sycl::cbrt(d0);
  d2 = cbrt(d0);
  // CHECK: d2 = sycl::cbrt((double)i);
  d2 = cbrt(i);

  // CHECK: d2 = sycl::ceil(d0);
  d2 = ceil(d0);
  // CHECK: d2 = sycl::ceil((double)i);
  d2 = ceil(i);

  // CHECK: d2 = sycl::copysign(d0, d1);
  d2 = copysign(d0, d1);
  // CHECK: d2 = sycl::copysign((double)i, (double)i);
  d2 = copysign(i, i);
  // CHECK: d2 = sycl::copysign(d0, (double)i);
  d2 = copysign(d0, i);
  // CHECK: d2 = sycl::copysign((double)i, d1);
  d2 = copysign(i, d1);

  // CHECK: d2 = sycl::cos(d0);
  d2 = cos(d0);
  // CHECK: d2 = sycl::cos((double)i);
  d2 = cos(i);

  // CHECK: d2 = sycl::cosh(d0);
  d2 = cosh(d0);
  // CHECK: d2 = sycl::cosh((double)i);
  d2 = cosh(i);

  // CHECK: d2 = sycl::cospi(d0);
  d2 = cospi(d0);
  // CHECK: d2 = sycl::cospi((double)i);
  d2 = cospi((double)i);

  // CHECK: d2 = sycl::erfc(d0);
  d2 = erfc(d0);
  // CHECK: d2 = sycl::erfc((double)i);
  d2 = erfc(i);

  // CHECK: d2 = sycl::erf(d0);
  d2 = erf(d0);
  // CHECK: d2 = sycl::erf((double)i);
  d2 = erf(i);

  // CHECK: d2 = sycl::exp10(d0);
  d2 = exp10(d0);
  // CHECK: d2 = sycl::exp10((double)i);
  d2 = exp10((double)i);

  // CHECK: d2 = sycl::exp2(d0);
  d2 = exp2(d0);
  // CHECK: d2 = sycl::exp2((double)i);
  d2 = exp2(i);

  // CHECK: d2 = sycl::exp(d0);
  d2 = exp(d0);
  // CHECK: d2 = sycl::exp((double)i);
  d2 = exp(i);

  // CHECK: d2 = sycl::expm1(d0);
  d2 = expm1(d0);
  // CHECK: d2 = sycl::expm1((double)i);
  d2 = expm1(i);

  // CHECK: d2 = sycl::cos(d0);
  d2 = cos(d0);
  // CHECK: d2 = sycl::cos((double)i);
  d2 = cos(i);

  // CHECK: d2 = sycl::cosh(d0);
  d2 = cosh(d0);
  // CHECK: d2 = sycl::cosh((double)i);
  d2 = cosh(i);

  // CHECK: d2 = sycl::cospi(d0);
  d2 = cospi(d0);
  // CHECK: d2 = sycl::cospi((double)i);
  d2 = cospi((double)i);

  // CHECK: d2 = sycl::erfc(d0);
  d2 = erfc(d0);
  // CHECK: d2 = sycl::erfc((double)i);
  d2 = erfc(i);

  // CHECK: d2 = sycl::erf(d0);
  d2 = erf(d0);
  // CHECK: d2 = sycl::erf((double)i);
  d2 = erf(i);

  // CHECK: d2 = sycl::exp10(d0);
  d2 = exp10(d0);
  // CHECK: d2 = sycl::exp10((double)i);
  d2 = exp10((double)i);

  // CHECK: d2 = sycl::exp2(d0);
  d2 = exp2(d0);
  // CHECK: d2 = sycl::exp2((double)i);
  d2 = exp2(i);

  // CHECK: d2 = sycl::exp(d0);
  d2 = exp(d0);
  // CHECK: d2 = sycl::exp((double)i);
  d2 = exp(i);

  // CHECK: d2 = sycl::expm1(d0);
  d2 = expm1(d0);
  // CHECK: d2 = sycl::expm1((double)i);
  d2 = expm1(i);

  // CHECK: d2 = sycl::fabs(d0);
  d2 = fabs(d0);
  // CHECK: d2 = sycl::fabs((double)i);
  d2 = fabs(i);

  // CHECK: sycl::fabs(d0);
  abs(d0);
  // CHECK: sycl::fabs(d0 * d1);
  abs(d0 * d1);

  // CHECK: d2 = sycl::fdim(d0, d1);
  d2 = fdim(d0, d1);
  // CHECK: d2 = sycl::fdim((double)i, (double)i);
  d2 = fdim(i, i);
  // CHECK: d2 = sycl::fdim(d0, (double)i);
  d2 = fdim(d0, i);
  // CHECK: d2 = sycl::fdim((double)i, d1);
  d2 = fdim(i, d1);

  // CHECK: d2 = sycl::floor(d0);
  d2 = floor(d0);
  // CHECK: d2 = sycl::floor((double)i);
  d2 = floor(i);

  // CHECK: d2 = sycl::fma(d0, d1, d2);
  d2 = fma(d0, d1, d2);
  // CHECK: d2 = sycl::fma((double)i, (double)i, (double)i);
  d2 = fma(i, i, i);
  // CHECK: d2 = sycl::fma(d0, (double)i, (double)i);
  d2 = fma(d0, i, i);
  // CHECK: d2 = sycl::fma((double)i, d1, (double)i);
  d2 = fma(i, d1, i);
  // CHECK: d2 = sycl::fma((double)i, (double)i, d2);
  d2 = fma(i, i, d2);
  // CHECK: d2 = sycl::fma(d0, d1, (double)i);
  d2 = fma(d0, d1, i);
  // CHECK: d2 = sycl::fma(d0, (double)i, d2);
  d2 = fma(d0, i, d2);
  // CHECK: d2 = sycl::fma((double)i, d1, d2);
  d2 = fma(i, d1, d2);

  // CHECK: d2 = sycl::fmax(d0, d1);
  d2 = fmax(d0, d1);
  // CHECK: d2 = sycl::fmax((double)i, (double)i);
  d2 = fmax(i, i);
  // CHECK: d2 = sycl::fmax(d0, (double)i);
  d2 = fmax(d0, i);
  // CHECK: d2 = sycl::fmax((double)i, d1);
  d2 = fmax(i, d1);

  // CHECK: d2 = sycl::fmin(d0, d1);
  d2 = fmin(d0, d1);
  // CHECK: d2 = sycl::fmin((double)i, (double)i);
  d2 = fmin(i, i);
  // CHECK: d2 = sycl::fmin(d0, (double)i);
  d2 = fmin(d0, i);
  // CHECK: d2 = sycl::fmin((double)i, d1);
  d2 = fmin(i, d1);

  // CHECK: d2 = sycl::fmod(d0, d1);
  d2 = fmod(d0, d1);
  // CHECK: d2 = sycl::fmod((double)i, (double)i);
  d2 = fmod(i, i);
  // CHECK: d2 = sycl::fmod(d0, (double)i);
  d2 = fmod(d0, i);
  // CHECK: d2 = sycl::fmod((double)i, d1);
  d2 = fmod(i, d1);

  // CHECK: d2 = sycl::frexp(d0, sycl::make_ptr<int, sycl::access::address_space::global_space>(&i));
  d2 = frexp(d0, &i);
  // CHECK: d2 = sycl::frexp((double)i, sycl::make_ptr<int, sycl::access::address_space::global_space>(&i));
  d2 = frexp(i, &i);

  // CHECK: d2 = sycl::hypot(d0, d1);
  d2 = hypot(d0, d1);
  // CHECK: d2 = sycl::hypot((double)i, (double)i);
  d2 = hypot(i, i);
  // CHECK: d2 = sycl::hypot(d0, (double)i);
  d2 = hypot(d0, i);
  // CHECK: d2 = sycl::hypot((double)i, d1);
  d2 = hypot(i, d1);

  // CHECK: d2 = sycl::ilogb(d0);
  d2 = ilogb(d0);
  // CHECK: d2 = sycl::ilogb((double)i);
  d2 = ilogb(i);

  // CHECK: d2 = sycl::ldexp(d0, i);
  d2 = ldexp(d0, i);
  // CHECK: d2 = sycl::ldexp((double)i, i);
  d2 = ldexp(i, i);

  // CHECK: d2 = sycl::lgamma(d0);
  d2 = lgamma(d0);
  // CHECK: d2 = sycl::lgamma((double)i);
  d2 = lgamma(i);

  // CHECK: d2 = sycl::rint(d0);
  d2 = llrint(d0);
  // CHECK: d2 = sycl::rint((double)i);
  d2 = llrint(i);

  // CHECK: d2 = sycl::round(d0);
  d2 = llround(d0);
  // CHECK: d2 = sycl::round((double)i);
  d2 = llround(i);

  // CHECK: d2 = sycl::log10(d0);
  d2 = log10(d0);
  // CHECK: d2 = sycl::log10((double)i);
  d2 = log10(i);

  // CHECK: d2 = sycl::log1p(d0);
  d2 = log1p(d0);
  // CHECK: d2 = sycl::log1p((double)i);
  d2 = log1p(i);

  // CHECK: d2 = sycl::log2(d0);
  d2 = log2(d0);
  // CHECK: d2 = sycl::log2((double)i);
  d2 = log2(i);

  // CHECK: d2 = sycl::logb(d0);
  d2 = logb(d0);
  // CHECK: d2 = sycl::logb((double)i);
  d2 = logb(i);

  // CHECK: d2 = sycl::rint(d0);
  d2 = lrint(d0);
  // CHECK: d2 = sycl::rint((double)i);
  d2 = lrint(i);

  // CHECK: d2 = sycl::round(d0);
  d2 = lround(d0);
  // CHECK: d2 = sycl::round((double)i);
  d2 = lround(i);

  // CHECK: d2 = sycl::modf(d0, sycl::make_ptr<double, sycl::access::address_space::global_space>(&d1));
  d2 = modf(d0, &d1);
  // CHECK: d2 = sycl::modf((double)i, sycl::make_ptr<double, sycl::access::address_space::global_space>(&d1));
  d2 = modf(i, &d1);

  // CHECK: d2 = sycl::nan(0u);
  d2 = nan("");

  // CHECK: d2 = sycl::pow(d0, d1);
  d2 = pow(d0, d1);
  // CHECK: d2 = sycl::pown((float)i, i);
  d2 = pow(i, i);
  // CHECK: d2 = sycl::pown(d0, i);
  d2 = pow(d0, i);
  // CHECK: d2 = sycl::pow((double)i, d1);
  d2 = pow(i, d1);

  // CHECK: sycl::pown(f, 1);
  float f;
  pow(f, 1);

  // CHECK: d2 = sycl::remainder(d0, d1);
  d2 = remainder(d0, d1);
  // CHECK: d2 = sycl::remainder((double)i, (double)i);
  d2 = remainder(i, i);
  // CHECK: d2 = sycl::remainder(d0, (double)i);
  d2 = remainder(d0, i);
  // CHECK: d2 = sycl::remainder((double)i, d1);
  d2 = remainder(i, d1);

  // CHECK: d2 = sycl::remquo(d0, d1, sycl::make_ptr<int, sycl::access::address_space::global_space>(&i));
  d2 = remquo(d0, d1, &i);
  // CHECK: d2 = sycl::remquo((double)i, (double)i, sycl::make_ptr<int, sycl::access::address_space::global_space>(&i));
  d2 = remquo(i, i, &i);
  // CHECK: d2 = sycl::remquo(d0, (double)i, sycl::make_ptr<int, sycl::access::address_space::global_space>(&i));
  d2 = remquo(d0, i, &i);
  // CHECK: d2 = sycl::remquo((double)i, d1, sycl::make_ptr<int, sycl::access::address_space::global_space>(&i));
  d2 = remquo(i, d1, &i);

  // CHECK: d2 = sycl::rint(d0);
  d2 = rint(d0);
  // CHECK: d2 = sycl::rint((double)i);
  d2 = rint(i);

  // CHECK: d2 = sycl::round(d0);
  d2 = round(d0);
  // CHECK: d2 = sycl::round((double)i);
  d2 = round(i);

  // CHECK: d2 = sycl::rsqrt(d0);
  d2 = rsqrt(d0);
  // CHECK: d2 = sycl::rsqrt((double)i);
  d2 = rsqrt((double)i);

  // CHECK: d1 = sycl::sincos(d0, sycl::make_ptr<double, sycl::access::address_space::global_space>(&d2));
  sincos(d0, &d1, &d2);
  // CHECK: d1 = sycl::sincos((double)i, sycl::make_ptr<double, sycl::access::address_space::global_space>(&d2));
  sincos(i, &d1, &d2);

  // CHECK: d2 = sycl::sin(d0);
  d2 = sin(d0);
  // CHECK: d2 = sycl::sin((double)i);
  d2 = sin(i);

  // CHECK: d2 = sycl::sinh(d0);
  d2 = sinh(d0);
  // CHECK: d2 = sycl::sinh((double)i);
  d2 = sinh(i);

  // CHECK: d2 = sycl::sinpi(d0);
  d2 = sinpi(d0);
  // CHECK: d2 = sycl::sinpi((double)i);
  d2 = sinpi((double)i);

  // CHECK: d2 = sycl::sqrt(d0);
  d2 = sqrt(d0);
  // CHECK: d2 = sycl::sqrt((double)i);
  d2 = sqrt(i);

  // CHECK: d2 = sycl::tan(d0);
  d2 = tan(d0);
  // CHECK: d2 = sycl::tan((double)i);
  d2 = tan(i);

  // CHECK: d2 = sycl::tanh(d0);
  d2 = tanh(d0);
  // CHECK: d2 = sycl::tanh((double)i);
  d2 = tanh(i);

  // CHECK: d2 = sycl::tgamma(d0);
  d2 = tgamma(d0);
  // CHECK: d2 = sycl::tgamma((double)i);
  d2 = tgamma(i);

  // CHECK: d2 = sycl::trunc(d0);
  d2 = trunc(d0);
  // CHECK: d2 = sycl::trunc((double)i);
  d2 = trunc(i);

  // CHECK: /*
  // CHECK-NEXT: DPCT1013:{{[0-9]+}}: The rounding mode could not be specified and the generated code may have different precision then the original code. Verify the correctness. SYCL math built-ins rounding mode is aligned with OpenCL C 1.2 standard.
  // CHECK-NEXT: */
  // CHECK-NEXT: d2 = d0 + d1;
  d2 = __dadd_rd(d0, d1);
  // CHECK: /*
  // CHECK-NEXT: DPCT1013:{{[0-9]+}}: The rounding mode could not be specified and the generated code may have different precision then the original code. Verify the correctness. SYCL math built-ins rounding mode is aligned with OpenCL C 1.2 standard.
  // CHECK-NEXT: */
  // CHECK-NEXT: d2 = d0 + d1;
  d2 = __dadd_rn(d0, d1);
  // CHECK: /*
  // CHECK-NEXT: DPCT1013:{{[0-9]+}}: The rounding mode could not be specified and the generated code may have different precision then the original code. Verify the correctness. SYCL math built-ins rounding mode is aligned with OpenCL C 1.2 standard.
  // CHECK-NEXT: */
  // CHECK-NEXT: d2 = d0 + d1;
  d2 = __dadd_ru(d0, d1);
  // CHECK: /*
  // CHECK-NEXT: DPCT1013:{{[0-9]+}}: The rounding mode could not be specified and the generated code may have different precision then the original code. Verify the correctness. SYCL math built-ins rounding mode is aligned with OpenCL C 1.2 standard.
  // CHECK-NEXT: */
  // CHECK-NEXT: d2 = d0 + d1;
  d2 = __dadd_rz(d0, d1);

  // CHECK: /*
  // CHECK-NEXT: DPCT1013:{{[0-9]+}}: The rounding mode could not be specified and the generated code may have different precision then the original code. Verify the correctness. SYCL math built-ins rounding mode is aligned with OpenCL C 1.2 standard.
  // CHECK-NEXT: */
  // CHECK-NEXT: d2 = d0 / d1;
  d2 = __ddiv_rd(d0, d1);
  // CHECK: /*
  // CHECK-NEXT: DPCT1013:{{[0-9]+}}: The rounding mode could not be specified and the generated code may have different precision then the original code. Verify the correctness. SYCL math built-ins rounding mode is aligned with OpenCL C 1.2 standard.
  // CHECK-NEXT: */
  // CHECK-NEXT: d2 = d0 / d1;
  d2 = __ddiv_rn(d0, d1);
  // CHECK: /*
  // CHECK-NEXT: DPCT1013:{{[0-9]+}}: The rounding mode could not be specified and the generated code may have different precision then the original code. Verify the correctness. SYCL math built-ins rounding mode is aligned with OpenCL C 1.2 standard.
  // CHECK-NEXT: */
  // CHECK-NEXT: d2 = d0 / d1;
  d2 = __ddiv_ru(d0, d1);
  // CHECK: /*
  // CHECK-NEXT: DPCT1013:{{[0-9]+}}: The rounding mode could not be specified and the generated code may have different precision then the original code. Verify the correctness. SYCL math built-ins rounding mode is aligned with OpenCL C 1.2 standard.
  // CHECK-NEXT: */
  // CHECK-NEXT: d2 = d0 / d1;
  d2 = __ddiv_rz(d0, d1);

  // CHECK: /*
  // CHECK-NEXT: DPCT1013:{{[0-9]+}}: The rounding mode could not be specified and the generated code may have different precision then the original code. Verify the correctness. SYCL math built-ins rounding mode is aligned with OpenCL C 1.2 standard.
  // CHECK-NEXT: */
  // CHECK-NEXT: d2 = d0 * d1;
  d2 = __dmul_rd(d0, d1);
  // CHECK: /*
  // CHECK-NEXT: DPCT1013:{{[0-9]+}}: The rounding mode could not be specified and the generated code may have different precision then the original code. Verify the correctness. SYCL math built-ins rounding mode is aligned with OpenCL C 1.2 standard.
  // CHECK-NEXT: */
  // CHECK-NEXT: d2 = d0 * d1;
  d2 = __dmul_rn(d0, d1);
  // CHECK: /*
  // CHECK-NEXT: DPCT1013:{{[0-9]+}}: The rounding mode could not be specified and the generated code may have different precision then the original code. Verify the correctness. SYCL math built-ins rounding mode is aligned with OpenCL C 1.2 standard.
  // CHECK-NEXT: */
  // CHECK-NEXT: d2 = d0 * d1;
  d2 = __dmul_ru(d0, d1);
  // CHECK: /*
  // CHECK-NEXT: DPCT1013:{{[0-9]+}}: The rounding mode could not be specified and the generated code may have different precision then the original code. Verify the correctness. SYCL math built-ins rounding mode is aligned with OpenCL C 1.2 standard.
  // CHECK-NEXT: */
  // CHECK-NEXT: d2 = d0 * d1;
  d2 = __dmul_rz(d0, d1);

  // CHECK: /*
  // CHECK-NEXT: DPCT1004:{{[0-9]+}}: Could not generate replacement.
  // CHECK-NEXT: */
  d1 = __drcp_rd(d0);
  // CHECK: /*
  // CHECK-NEXT: DPCT1004:{{[0-9]+}}: Could not generate replacement.
  // CHECK-NEXT: */
  d1 = __drcp_rn(d0);
  // CHECK: /*
  // CHECK-NEXT: DPCT1004:{{[0-9]+}}: Could not generate replacement.
  // CHECK-NEXT: */
  d1 = __drcp_ru(d0);
  // CHECK: /*
  // CHECK-NEXT: DPCT1004:{{[0-9]+}}: Could not generate replacement.
  // CHECK-NEXT: */
  d1 = __drcp_rz(d0);

  // CHECK: /*
  // CHECK-NEXT: DPCT1013:{{[0-9]+}}: The rounding mode could not be specified and the generated code may have different precision then the original code. Verify the correctness. SYCL math built-ins rounding mode is aligned with OpenCL C 1.2 standard.
  // CHECK-NEXT: */
  // CHECK-NEXT: d0 = sycl::sqrt(d0);
  d0 = __dsqrt_rd(d0);
  // CHECK: /*
  // CHECK-NEXT: DPCT1013:{{[0-9]+}}: The rounding mode could not be specified and the generated code may have different precision then the original code. Verify the correctness. SYCL math built-ins rounding mode is aligned with OpenCL C 1.2 standard.
  // CHECK-NEXT: */
  // CHECK-NEXT: d1 = sycl::sqrt(d1);
  d1 = __dsqrt_rn(d1);
  // CHECK: /*
  // CHECK-NEXT: DPCT1013:{{[0-9]+}}: The rounding mode could not be specified and the generated code may have different precision then the original code. Verify the correctness. SYCL math built-ins rounding mode is aligned with OpenCL C 1.2 standard.
  // CHECK-NEXT: */
  // CHECK-NEXT: d0 = sycl::sqrt(d0);
  d0 = __dsqrt_ru(d0);
  // CHECK: /*
  // CHECK-NEXT: DPCT1013:{{[0-9]+}}: The rounding mode could not be specified and the generated code may have different precision then the original code. Verify the correctness. SYCL math built-ins rounding mode is aligned with OpenCL C 1.2 standard.
  // CHECK-NEXT: */
  // CHECK-NEXT: d1 = sycl::sqrt(d1);
  d1 = __dsqrt_rz(d1);

  // CHECK: /*
  // CHECK-NEXT: DPCT1013:{{[0-9]+}}: The rounding mode could not be specified and the generated code may have different precision then the original code. Verify the correctness. SYCL math built-ins rounding mode is aligned with OpenCL C 1.2 standard.
  // CHECK-NEXT: */
  // CHECK-NEXT: d0 = sycl::sqrt((double)i);
  d0 = __dsqrt_rd(i);
  // CHECK: /*
  // CHECK-NEXT: DPCT1013:{{[0-9]+}}: The rounding mode could not be specified and the generated code may have different precision then the original code. Verify the correctness. SYCL math built-ins rounding mode is aligned with OpenCL C 1.2 standard.
  // CHECK-NEXT: */
  // CHECK-NEXT: d1 = sycl::sqrt((double)i);
  d1 = __dsqrt_rn(i);
  // CHECK: /*
  // CHECK-NEXT: DPCT1013:{{[0-9]+}}: The rounding mode could not be specified and the generated code may have different precision then the original code. Verify the correctness. SYCL math built-ins rounding mode is aligned with OpenCL C 1.2 standard.
  // CHECK-NEXT: */
  // CHECK-NEXT: d0 = sycl::sqrt((double)i);
  d0 = __dsqrt_ru(i);
  // CHECK: /*
  // CHECK-NEXT: DPCT1013:{{[0-9]+}}: The rounding mode could not be specified and the generated code may have different precision then the original code. Verify the correctness. SYCL math built-ins rounding mode is aligned with OpenCL C 1.2 standard.
  // CHECK-NEXT: */
  // CHECK-NEXT: d1 = sycl::sqrt((double)i);
  d1 = __dsqrt_rz(i);

  // CHECK: /*
  // CHECK-NEXT: DPCT1013:{{[0-9]+}}: The rounding mode could not be specified and the generated code may have different precision then the original code. Verify the correctness. SYCL math built-ins rounding mode is aligned with OpenCL C 1.2 standard.
  // CHECK-NEXT: */
  // CHECK-NEXT: d2 = d0 - d1;
  d2 = __dsub_rd(d0, d1);
  // CHECK: /*
  // CHECK-NEXT: DPCT1013:{{[0-9]+}}: The rounding mode could not be specified and the generated code may have different precision then the original code. Verify the correctness. SYCL math built-ins rounding mode is aligned with OpenCL C 1.2 standard.
  // CHECK-NEXT: */
  // CHECK-NEXT: d2 = d0 - d1;
  d2 = __dsub_rn(d0, d1);
  // CHECK: /*
  // CHECK-NEXT: DPCT1013:{{[0-9]+}}: The rounding mode could not be specified and the generated code may have different precision then the original code. Verify the correctness. SYCL math built-ins rounding mode is aligned with OpenCL C 1.2 standard.
  // CHECK-NEXT: */
  // CHECK-NEXT: d2 = d0 - d1;
  d2 = __dsub_ru(d0, d1);
  // CHECK: /*
  // CHECK-NEXT: DPCT1013:{{[0-9]+}}: The rounding mode could not be specified and the generated code may have different precision then the original code. Verify the correctness. SYCL math built-ins rounding mode is aligned with OpenCL C 1.2 standard.
  // CHECK-NEXT: */
  // CHECK-NEXT: d2 = d0 - d1;
  d2 = __dsub_rz(d0, d1);

  // CHECK: /*
  // CHECK-NEXT: DPCT1013:{{[0-9]+}}: The rounding mode could not be specified and the generated code may have different precision then the original code. Verify the correctness. SYCL math built-ins rounding mode is aligned with OpenCL C 1.2 standard.
  // CHECK-NEXT: */
  // CHECK-NEXT: d2 = sycl::fma(d0, d1, d2);
  d2 = __fma_rd(d0, d1, d2);
  // CHECK: /*
  // CHECK-NEXT: DPCT1013:{{[0-9]+}}: The rounding mode could not be specified and the generated code may have different precision then the original code. Verify the correctness. SYCL math built-ins rounding mode is aligned with OpenCL C 1.2 standard.
  // CHECK-NEXT: */
  // CHECK-NEXT: d2 = sycl::fma(d0, d1, d2);
  d2 = __fma_rn(d0, d1, d2);
  // CHECK: /*
  // CHECK-NEXT: DPCT1013:{{[0-9]+}}: The rounding mode could not be specified and the generated code may have different precision then the original code. Verify the correctness. SYCL math built-ins rounding mode is aligned with OpenCL C 1.2 standard.
  // CHECK-NEXT: */
  // CHECK-NEXT: d2 = sycl::fma(d0, d1, d2);
  d2 = __fma_ru(d0, d1, d2);
  // CHECK: /*
  // CHECK-NEXT: DPCT1013:{{[0-9]+}}: The rounding mode could not be specified and the generated code may have different precision then the original code. Verify the correctness. SYCL math built-ins rounding mode is aligned with OpenCL C 1.2 standard.
  // CHECK-NEXT: */
  // CHECK-NEXT: d2 = sycl::fma(d0, d1, d2);
  d2 = __fma_rz(d0, d1, d2);

  // CHECK: /*
  // CHECK-NEXT: DPCT1013:{{[0-9]+}}: The rounding mode could not be specified and the generated code may have different precision then the original code. Verify the correctness. SYCL math built-ins rounding mode is aligned with OpenCL C 1.2 standard.
  // CHECK-NEXT: */
  // CHECK-NEXT: d2 = sycl::fma((double)i, (double)i, (double)i);
  d2 = __fma_rd(i, i, i);
  // CHECK: /*
  // CHECK-NEXT: DPCT1013:{{[0-9]+}}: The rounding mode could not be specified and the generated code may have different precision then the original code. Verify the correctness. SYCL math built-ins rounding mode is aligned with OpenCL C 1.2 standard.
  // CHECK-NEXT: */
  // CHECK-NEXT: d2 = sycl::fma((double)i, (double)i, (double)i);
  d2 = __fma_rn(i, i, i);
  // CHECK: /*
  // CHECK-NEXT: DPCT1013:{{[0-9]+}}: The rounding mode could not be specified and the generated code may have different precision then the original code. Verify the correctness. SYCL math built-ins rounding mode is aligned with OpenCL C 1.2 standard.
  // CHECK-NEXT: */
  // CHECK-NEXT: d2 = sycl::fma((double)i, (double)i, (double)i);
  d2 = __fma_ru(i, i, i);
  // CHECK: /*
  // CHECK-NEXT: DPCT1013:{{[0-9]+}}: The rounding mode could not be specified and the generated code may have different precision then the original code. Verify the correctness. SYCL math built-ins rounding mode is aligned with OpenCL C 1.2 standard.
  // CHECK-NEXT: */
  // CHECK-NEXT: d2 = sycl::fma((double)i, (double)i, (double)i);
  d2 = __fma_rz(i, i, i);

  // CHECK: d0 = sycl::fmin(d0, d1);
  d0 = fmin(d0, d1);
  // CHECK: d0 = sycl::fmin((double)i, (double)i);
  d0 = fmin(i, i);
  // CHECK: d0 = sycl::fmin(d0, (double)i);
  d0 = fmin(d0, i);
  // CHECK: d0 = sycl::fmin((double)i, d1);
  d0 = fmin(i, d1);

  // CHECK: d0 = sycl::fmax(d0, d1);
  d0 = fmax(d0, d1);
  // CHECK: d0 = sycl::fmax((double)i, (double)i);
  d0 = fmax(i, i);
  // CHECK: d0 = sycl::fmax(d0, (double)i);
  d0 = fmax(d0, i);
  // CHECK: d0 = sycl::fmax((double)i, d1);
  d0 = fmax(i, d1);

  // CHECK: d1 = sycl::floor(d1);
  d1 = floor(d1);
  // CHECK: d1 = sycl::floor((double)i);
  d1 = floor(i);

  // CHECK: d2 = sycl::ceil(d2);
  d2 = ceil(d2);
  // CHECK: d2 = sycl::ceil((double)i);
  d2 = ceil(i);

  // CHECK: d2 = sycl::fma(d0, d1, d2);
  d2 = fma(d0, d1, d2);
  // CHECK: d2 = sycl::fma((double)i, (double)i, (double)i);
  d2 = fma(i, i, i);
  // CHECK: d2 = sycl::fma(d0, (double)i, (double)i);
  d2 = fma(d0, i, i);
  // CHECK: d2 = sycl::fma((double)i, d1, (double)i);
  d2 = fma(i, d1, i);
  // CHECK: d2 = sycl::fma((double)i, (double)i, d2);
  d2 = fma(i, i, d2);
  // CHECK: d2 = sycl::fma(d0, d1, (double)i);
  d2 = fma(d0, d1, i);
  // CHECK: d2 = sycl::fma(d0, (double)i, d2);
  d2 = fma(d0, i, d2);
  // CHECK: d2 = sycl::fma((double)i, d1, d2);
  d2 = fma(i, d1, d2);

  // CHECK: d2 = sycl::nan(0u);
  d2 = nan("NaN");

  // CHECK: d0 = sycl::nextafter(d0, d0);
  d0 = nextafter(d0, d0);
  // CHECK: d0 = sycl::nextafter((double)i, (double)i);
  d0 = nextafter(i, i);
  // CHECK: d0 = sycl::nextafter(d0, (double)i);
  d0 = nextafter(d0, i);
  // CHECK: d0 = sycl::nextafter((double)i, d1);
  d0 = nextafter(i, d1);
}

__global__ void kernelFuncFloat(float *deviceArrayFloat) {
  float &f0 = *deviceArrayFloat, &f1 = *(deviceArrayFloat + 1), &f2 = *(deviceArrayFloat + 2);
  int i;

  // Single Precision Mathematical Functions

  // CHECK: f2 = sycl::log(f0);
  f2 = logf(f0);
  // CHECK: f2 = sycl::log((float)i);
  f2 = logf(i);

  // CHECK: f2 = sycl::acos(f0);
  f2 = acosf(f0);
  // CHECK: f2 = sycl::acos((float)i);
  f2 = acosf(i);

  // CHECK: f2 = sycl::acosh(f0);
  f2 = acoshf(f0);
  // CHECK: f2 = sycl::acosh((float)i);
  f2 = acoshf(i);

  // CHECK: f2 = sycl::asin(f0);
  f2 = asinf(f0);
  // CHECK: f2 = sycl::asin((float)i);
  f2 = asinf(i);

  // CHECK: f2 = sycl::asinh(f0);
  f2 = asinhf(f0);
  // CHECK: f2 = sycl::asinh((float)i);
  f2 = asinhf(i);

  // CHECK: f2 = sycl::atan2(f0, f1);
  f2 = atan2f(f0, f1);
  // CHECK: f2 = sycl::atan2((float)i, (float)i);
  f2 = atan2f(i, i);
  // CHECK: f2 = sycl::atan2(f0, (float)i);
  f2 = atan2f(f0, i);
  // CHECK: f2 = sycl::atan2((float)i, f1);
  f2 = atan2f(i, f1);

  // CHECK: f2 = sycl::atan(f0);
  f2 = atanf(f0);
  // CHECK: f2 = sycl::atan((float)i);
  f2 = atanf(i);

  // CHECK: f2 = sycl::atanh(f0);
  f2 = atanhf(f0);
  // CHECK: f2 = sycl::atanh((float)i);
  f2 = atanhf(i);

  // CHECK: f2 = sycl::cbrt(f0);
  f2 = cbrtf(f0);
  // CHECK: f2 = sycl::cbrt((float)i);
  f2 = cbrtf(i);

  // CHECK: f2 = sycl::ceil(f0);
  f2 = ceilf(f0);
  // CHECK: f2 = sycl::ceil((float)i);
  f2 = ceilf(i);

  // CHECK: f2 = sycl::copysign(f0, f1);
  f2 = copysignf(f0, f1);
  // CHECK: f2 = sycl::copysign((float)i, (float)i);
  f2 = copysignf(i, i);
  // CHECK: f2 = sycl::copysign(f0, (float)i);
  f2 = copysignf(f0, i);
  // CHECK: f2 = sycl::copysign((float)i, f1);
  f2 = copysignf(i, f1);

  // CHECK: f2 = sycl::cos(f0);
  f2 = cosf(f0);
  // CHECK: f2 = sycl::cos((float)i);
  f2 = cosf(i);

  // CHECK: f2 = sycl::cosh(f0);
  f2 = coshf(f0);
  // CHECK: f2 = sycl::cosh((float)i);
  f2 = coshf(i);

  // CHECK: f2 = sycl::cospi(f0);
  f2 = cospif(f0);
  // CHECK: f2 = sycl::cospi((float)i);
  f2 = cospif(i);

  // CHECK: f2 = sycl::erfc(f0);
  f2 = erfcf(f0);
  // CHECK: f2 = sycl::erfc((float)i);
  f2 = erfcf(i);

  // CHECK: f2 = sycl::erf(f0);
  f2 = erff(f0);
  // CHECK: f2 = sycl::erf((float)i);
  f2 = erff(i);

  // CHECK: f2 = sycl::exp10(f0);
  f2 = exp10f(f0);
  // CHECK: f2 = sycl::exp10((float)i);
  f2 = exp10f(i);

  // CHECK: f2 = sycl::exp2(f0);
  f2 = exp2f(f0);
  // CHECK: f2 = sycl::exp2((float)i);
  f2 = exp2f(i);

  // CHECK: f2 = sycl::exp(f0);
  f2 = expf(f0);
  // CHECK: f2 = sycl::exp((float)i);
  f2 = expf(i);

  // CHECK: f2 = sycl::expm1(f0);
  f2 = expm1f(f0);
  // CHECK: f2 = sycl::expm1((float)i);
  f2 = expm1f(i);

  // CHECK: f2 = sycl::fabs(f0);
  f2 = fabsf(f0);
  // CHECK: f2 = sycl::fabs((float)i);
  f2 = fabsf(i);

  // CHECK: f2 = sycl::fdim(f0, f1);
  f2 = fdimf(f0, f1);
  // CHECK: f2 = sycl::fdim((float)i, (float)i);
  f2 = fdimf(i, i);
  // CHECK: f2 = sycl::fdim(f0, (float)i);
  f2 = fdimf(f0, i);
  // CHECK: f2 = sycl::fdim((float)i, f1);
  f2 = fdimf(i, f1);

  // CHECK: f2 = sycl::native::divide(f0, f1);
  f2 = fdividef(f0, f1);
  // CHECK: f2 = sycl::native::divide((float)i, (float)i);
  f2 = fdividef(i, i);
  // CHECK: f2 = sycl::native::divide(f0, (float)i);
  f2 = fdividef(f0, i);
  // CHECK: f2 = sycl::native::divide((float)i, f1);
  f2 = fdividef(i, f1);

  // CHECK: f2 = sycl::floor(f0);
  f2 = floorf(f0);
  // CHECK: f2 = sycl::floor((float)i);
  f2 = floorf(i);

  // CHECK: f2 = sycl::fma(f0, f1, f2);
  f2 = fmaf(f0, f1, f2);
  // CHECK: f2 = sycl::fma((float)i, (float)i, (float)i);
  f2 = fmaf(i, i, i);
  // CHECK: f2 = sycl::fma(f0, (float)i, (float)i);
  f2 = fmaf(f0, i, i);
  // CHECK: f2 = sycl::fma((float)i, f1, (float)i);
  f2 = fmaf(i, f1, i);
  // CHECK: f2 = sycl::fma((float)i, (float)i, f2);
  f2 = fmaf(i, i, f2);
  // CHECK: f2 = sycl::fma(f0, f1, (float)i);
  f2 = fmaf(f0, f1, i);
  // CHECK: f2 = sycl::fma(f0, (float)i, f2);
  f2 = fmaf(f0, i, f2);
  // CHECK: f2 = sycl::fma((float)i, f1, f2);
  f2 = fmaf(i, f1, f2);

  // CHECK: f2 = sycl::fmax(f0, f1);
  f2 = fmaxf(f0, f1);
  // CHECK: f2 = sycl::fmax((float)i, (float)i);
  f2 = fmaxf(i, i);
  // CHECK: f2 = sycl::fmax(f0, (float)i);
  f2 = fmaxf(f0, i);
  // CHECK: f2 = sycl::fmax((float)i, f1);
  f2 = fmaxf(i, f1);

  // CHECK: f2 = sycl::fmin(f0, f1);
  f2 = fminf(f0, f1);
  // CHECK: f2 = sycl::fmin((float)i, (float)i);
  f2 = fminf(i, i);
  // CHECK: f2 = sycl::fmin(f0, (float)i);
  f2 = fminf(f0, i);
  // CHECK: f2 = sycl::fmin((float)i, f1);
  f2 = fminf(i, f1);

  // CHECK: f2 = sycl::fmod(f0, f1);
  f2 = fmodf(f0, f1);
  // CHECK: f2 = sycl::fmod((float)i, (float)i);
  f2 = fmodf(i, i);
  // CHECK: f2 = sycl::fmod(f0, (float)i);
  f2 = fmodf(f0, i);
  // CHECK: f2 = sycl::fmod((float)i, f1);
  f2 = fmodf(i, f1);

  // CHECK: f2 = sycl::frexp(f0, sycl::make_ptr<int, sycl::access::address_space::global_space>(&i));
  f2 = frexpf(f0, &i);
  // CHECK: f2 = sycl::frexp((float)i, sycl::make_ptr<int, sycl::access::address_space::global_space>(&i));
  f2 = frexpf(i, &i);

  // CHECK: f2 = sycl::hypot(f0, f1);
  f2 = hypotf(f0, f1);
  // CHECK: f2 = sycl::hypot((float)i, (float)i);
  f2 = hypotf(i, i);
  // CHECK: f2 = sycl::hypot(f0, (float)i);
  f2 = hypotf(f0, i);
  // CHECK: f2 = sycl::hypot((float)i, f1);
  f2 = hypotf(i, f1);

  // CHECK: f2 = sycl::ilogb(f0);
  f2 = ilogbf(f0);
  // CHECK: f2 = sycl::ilogb((float)i);
  f2 = ilogbf(i);

  // CHECK: i = sycl::isfinite(f0);
  i = isfinite(f0);
  // CHECK: i = sycl::isfinite((float)i);
  i = isfinite(i);

  // CHECK: i = sycl::isinf(f0);
  i = isinf(f0);
  // CHECK: i = sycl::isinf((float)i);
  i = isinf(i);

  // CHECK: i = sycl::isnan(f0);
  i = isnan(f0);
  // CHECK: i = sycl::isnan((float)i);
  i = isnan(i);

  // CHECK: f2 = sycl::ldexp(f0, i);
  f2 = ldexpf(f0, i);
  // CHECK: f2 = sycl::ldexp((float)i, i);
  f2 = ldexpf(i, i);

  // CHECK: f2 = sycl::lgamma(f0);
  f2 = lgammaf(f0);
  // CHECK: f2 = sycl::lgamma((float)i);
  f2 = lgammaf(i);

  // CHECK: f2 = sycl::rint(f0);
  f2 = llrintf(f0);
  // CHECK: f2 = sycl::rint((float)i);
  f2 = llrintf(i);

  // CHECK: f2 = sycl::round(f0);
  f2 = llroundf(f0);
  // CHECK: f2 = sycl::round((float)i);
  f2 = llroundf(i);

  // CHECK: f2 = sycl::log10(f0);
  f2 = log10f(f0);
  // CHECK: f2 = sycl::log10((float)i);
  f2 = log10f(i);

  // CHECK: f2 = sycl::log1p(f0);
  f2 = log1pf(f0);
  // CHECK: f2 = sycl::log1p((float)i);
  f2 = log1pf(i);

  // CHECK: f2 = sycl::log2(f0);
  f2 = log2f(f0);
  // CHECK: f2 = sycl::log2((float)i);
  f2 = log2f(i);

  // CHECK: f2 = sycl::logb(f0);
  f2 = logbf(f0);
  // CHECK: f2 = sycl::logb((float)i);
  f2 = logbf(i);

  // CHECK: f2 = sycl::rint(f0);
  f2 = lrintf(f0);
  // CHECK: f2 = sycl::rint((float)i);
  f2 = lrintf(i);

  // CHECK: f2 = sycl::round(f0);
  f2 = lroundf(f0);
  // CHECK: f2 = sycl::round((float)i);
  f2 = lroundf(i);

  // CHECK: f2 = sycl::modf(f0, sycl::make_ptr<float, sycl::access::address_space::global_space>(&f1));
  f2 = modff(f0, &f1);
  // CHECK: f2 = sycl::modf((float)i, sycl::make_ptr<float, sycl::access::address_space::global_space>(&f1));
  f2 = modff(i, &f1);

  // CHECK: f2 = sycl::nan(0u);
  f2 = nan("");

  // CHECK: f2 = sycl::pow(f0, f1);
  f2 = powf(f0, f1);
  // CHECK: f2 = sycl::pow((float)i, (float)i);
  f2 = powf(i, i);
  // CHECK: f2 = sycl::pow(f0, (float)i);
  f2 = powf(f0, i);
  // CHECK: f2 = sycl::pow((float)i, f1);
  f2 = powf(i, f1);

  // CHECK: f2 = sycl::remainder(f0, f1);
  f2 = remainderf(f0, f1);
  // CHECK: f2 = sycl::remainder((float)i, (float)i);
  f2 = remainderf(i, i);
  // CHECK: f2 = sycl::remainder(f0, (float)i);
  f2 = remainderf(f0, i);
  // CHECK: f2 = sycl::remainder((float)i, f1);
  f2 = remainderf(i, f1);

  // CHECK: f2 = sycl::remquo(f0, f1, sycl::make_ptr<int, sycl::access::address_space::global_space>(&i));
  f2 = remquof(f0, f1, &i);
  // CHECK: f2 = sycl::remquo((float)i, (float)i, sycl::make_ptr<int, sycl::access::address_space::global_space>(&i));
  f2 = remquof(i, i, &i);
  // CHECK: f2 = sycl::remquo(f0, (float)i, sycl::make_ptr<int, sycl::access::address_space::global_space>(&i));
  f2 = remquof(f0, i, &i);
  // CHECK: f2 = sycl::remquo((float)i, f1, sycl::make_ptr<int, sycl::access::address_space::global_space>(&i));
  f2 = remquof(i, f1, &i);

  // CHECK: f2 = sycl::rint(f0);
  f2 = rintf(f0);
  // CHECK: f2 = sycl::rint((float)i);
  f2 = rintf(i);

  // CHECK: f2 = sycl::round(f0);
  f2 = roundf(f0);
  // CHECK: f2 = sycl::round((float)i);
  f2 = roundf(i);

  // CHECK: f2 = sycl::rsqrt(f0);
  f2 = rsqrtf(f0);
  // CHECK: f2 = sycl::rsqrt((float)i);
  f2 = rsqrtf(i);

  // CHECK: f2 = sycl::signbit(f0);
  f2 = signbit(f0);
  // CHECK: f2 = sycl::signbit((float)i);
  f2 = signbit(i);

  // CHECK: f1 = sycl::sincos(f0, sycl::make_ptr<float, sycl::access::address_space::global_space>(&f2));
  sincosf(f0, &f1, &f2);
  // CHECK: f1 = sycl::sincos((float)i, sycl::make_ptr<float, sycl::access::address_space::global_space>(&f2));
  sincosf(i, &f1, &f2);

  // CHECK: f2 = sycl::sin(f0);
  f2 = sinf(f0);
  // CHECK: f2 = sycl::sin((float)i);
  f2 = sinf(i);

  // CHECK: f2 = sycl::sinh(f0);
  f2 = sinhf(f0);
  // CHECK: f2 = sycl::sinh((float)i);
  f2 = sinhf(i);

  // CHECK: f2 = sycl::sinpi(f0);
  f2 = sinpif(f0);
  // CHECK: f2 = sycl::sinpi((float)i);
  f2 = sinpif(i);

  // CHECK: f2 = sycl::sqrt(f0);
  f2 = sqrtf(f0);
  // CHECK: f2 = sycl::sqrt((float)i);
  f2 = sqrtf(i);

  // CHECK: f2 = sycl::tan(f0);
  f2 = tanf(f0);
  // CHECK: f2 = sycl::tan((float)i);
  f2 = tanf(i);

  // CHECK: f2 = sycl::tanh(f0);
  f2 = tanhf(f0);
  // CHECK: f2 = sycl::tanh((float)i);
  f2 = tanhf(i);

  // CHECK: f2 = sycl::tgamma(f0);
  f2 = tgammaf(f0);
  // CHECK: f2 = sycl::tgamma((float)i);
  f2 = tgammaf(i);

  // CHECK: f2 = sycl::trunc(f0);
  f2 = truncf(f0);
  // CHECK: f2 = sycl::trunc((float)i);
  f2 = truncf(i);

  // CHECK: f0 = sycl::cos(f0);
  f0 = __cosf(f0);
  // CHECK: f0 = sycl::cos((float)i);
  f0 = __cosf(i);

  // CHECK: f0 = sycl::exp10(f0);
  f0 = __exp10f(f0);
  // CHECK: f0 = sycl::exp10((float)i);
  f0 = __exp10f(i);

  // CHECK: f0 = sycl::exp(f0);
  f0 = __expf(f0);
  // CHECK: f0 = sycl::exp((float)i);
  f0 = __expf(i);

  // CHECK: /*
  // CHECK-NEXT: DPCT1013:{{[0-9]+}}: The rounding mode could not be specified and the generated code may have different precision then the original code. Verify the correctness. SYCL math built-ins rounding mode is aligned with OpenCL C 1.2 standard.
  // CHECK-NEXT: */
  // CHECK-NEXT: f2 = f0 + f1;
  f2 = __fadd_rd(f0, f1);
  // CHECK: /*
  // CHECK-NEXT: DPCT1013:{{[0-9]+}}: The rounding mode could not be specified and the generated code may have different precision then the original code. Verify the correctness. SYCL math built-ins rounding mode is aligned with OpenCL C 1.2 standard.
  // CHECK-NEXT: */
  // CHECK-NEXT: f2 = f0 + f1;
  f2 = __fadd_rn(f0, f1);
  // CHECK: /*
  // CHECK-NEXT: DPCT1013:{{[0-9]+}}: The rounding mode could not be specified and the generated code may have different precision then the original code. Verify the correctness. SYCL math built-ins rounding mode is aligned with OpenCL C 1.2 standard.
  // CHECK-NEXT: */
  // CHECK-NEXT: f2 = f0 + f1;
  f2 = __fadd_ru(f0, f1);
  // CHECK: /*
  // CHECK-NEXT: DPCT1013:{{[0-9]+}}: The rounding mode could not be specified and the generated code may have different precision then the original code. Verify the correctness. SYCL math built-ins rounding mode is aligned with OpenCL C 1.2 standard.
  // CHECK-NEXT: */
  // CHECK-NEXT: f2 = f0 + f1;
  f2 = __fadd_rz(f0, f1);

  // CHECK: /*
  // CHECK-NEXT: DPCT1013:{{[0-9]+}}: The rounding mode could not be specified and the generated code may have different precision then the original code. Verify the correctness. SYCL math built-ins rounding mode is aligned with OpenCL C 1.2 standard.
  // CHECK-NEXT: */
  // CHECK-NEXT: f2 = f0 / f1;
  f2 = __fdiv_rd(f0, f1);
  // CHECK: /*
  // CHECK-NEXT: DPCT1013:{{[0-9]+}}: The rounding mode could not be specified and the generated code may have different precision then the original code. Verify the correctness. SYCL math built-ins rounding mode is aligned with OpenCL C 1.2 standard.
  // CHECK-NEXT: */
  // CHECK-NEXT: f2 = f0 / f1;
  f2 = __fdiv_rn(f0, f1);
  // CHECK: /*
  // CHECK-NEXT: DPCT1013:{{[0-9]+}}: The rounding mode could not be specified and the generated code may have different precision then the original code. Verify the correctness. SYCL math built-ins rounding mode is aligned with OpenCL C 1.2 standard.
  // CHECK-NEXT: */
  // CHECK-NEXT: f2 = f0 / f1;
  f2 = __fdiv_ru(f0, f1);
  // CHECK: /*
  // CHECK-NEXT: DPCT1013:{{[0-9]+}}: The rounding mode could not be specified and the generated code may have different precision then the original code. Verify the correctness. SYCL math built-ins rounding mode is aligned with OpenCL C 1.2 standard.
  // CHECK-NEXT: */
  // CHECK-NEXT: f2 = f0 / f1;
  f2 = __fdiv_rz(f0, f1);

  // CHECK: f2 = sycl::native::divide(f0, f1);
  f2 = __fdividef(f0, f1);
  // CHECK: f2 = sycl::native::divide((float)i, (float)i);
  f2 = __fdividef(i, i);
  // CHECK: f2 = sycl::native::divide(f0, (float)i);
  f2 = __fdividef(f0, i);
  // CHECK: f2 = sycl::native::divide((float)i, f1);
  f2 = __fdividef(i, f1);

  // CHECK: /*
  // CHECK-NEXT: DPCT1013:{{[0-9]+}}: The rounding mode could not be specified and the generated code may have different precision then the original code. Verify the correctness. SYCL math built-ins rounding mode is aligned with OpenCL C 1.2 standard.
  // CHECK-NEXT: */
  // CHECK-NEXT: f2 = sycl::fma(f0, f1, f2);
  f2 = __fmaf_rd(f0, f1, f2);
  // CHECK: /*
  // CHECK-NEXT: DPCT1013:{{[0-9]+}}: The rounding mode could not be specified and the generated code may have different precision then the original code. Verify the correctness. SYCL math built-ins rounding mode is aligned with OpenCL C 1.2 standard.
  // CHECK-NEXT: */
  // CHECK-NEXT: f2 = sycl::fma(f0, f1, f2);
  f2 = __fmaf_rn(f0, f1, f2);
  // CHECK: /*
  // CHECK-NEXT: DPCT1013:{{[0-9]+}}: The rounding mode could not be specified and the generated code may have different precision then the original code. Verify the correctness. SYCL math built-ins rounding mode is aligned with OpenCL C 1.2 standard.
  // CHECK-NEXT: */
  // CHECK-NEXT: f2 = sycl::fma(f0, f1, f2);
  f2 = __fmaf_ru(f0, f1, f2);
  // CHECK: /*
  // CHECK-NEXT: DPCT1013:{{[0-9]+}}: The rounding mode could not be specified and the generated code may have different precision then the original code. Verify the correctness. SYCL math built-ins rounding mode is aligned with OpenCL C 1.2 standard.
  // CHECK-NEXT: */
  // CHECK-NEXT: f2 = sycl::fma(f0, f1, f2);
  f2 = __fmaf_rz(f0, f1, f2);

  // CHECK: /*
  // CHECK-NEXT: DPCT1013:{{[0-9]+}}: The rounding mode could not be specified and the generated code may have different precision then the original code. Verify the correctness. SYCL math built-ins rounding mode is aligned with OpenCL C 1.2 standard.
  // CHECK-NEXT: */
  // CHECK-NEXT: f2 = sycl::fma((float)i, (float)i, (float)i);
  f2 = __fmaf_rd(i, i, i);
  // CHECK: /*
  // CHECK-NEXT: DPCT1013:{{[0-9]+}}: The rounding mode could not be specified and the generated code may have different precision then the original code. Verify the correctness. SYCL math built-ins rounding mode is aligned with OpenCL C 1.2 standard.
  // CHECK-NEXT: */
  // CHECK-NEXT: f2 = sycl::fma((float)i, (float)i, (float)i);
  f2 = __fmaf_rn(i, i, i);
  // CHECK: /*
  // CHECK-NEXT: DPCT1013:{{[0-9]+}}: The rounding mode could not be specified and the generated code may have different precision then the original code. Verify the correctness. SYCL math built-ins rounding mode is aligned with OpenCL C 1.2 standard.
  // CHECK-NEXT: */
  // CHECK-NEXT: f2 = sycl::fma((float)i, (float)i, (float)i);
  f2 = __fmaf_ru(i, i, i);
  // CHECK: /*
  // CHECK-NEXT: DPCT1013:{{[0-9]+}}: The rounding mode could not be specified and the generated code may have different precision then the original code. Verify the correctness. SYCL math built-ins rounding mode is aligned with OpenCL C 1.2 standard.
  // CHECK-NEXT: */
  // CHECK-NEXT: f2 = sycl::fma((float)i, (float)i, (float)i);
  f2 = __fmaf_rz(i, i, i);

  // CHECK: /*
  // CHECK-NEXT: DPCT1013:{{[0-9]+}}: The rounding mode could not be specified and the generated code may have different precision then the original code. Verify the correctness. SYCL math built-ins rounding mode is aligned with OpenCL C 1.2 standard.
  // CHECK-NEXT: */
  // CHECK: f2 = f0 * f1;
  f2 = __fmul_rd(f0, f1);
  // CHECK: /*
  // CHECK-NEXT: DPCT1013:{{[0-9]+}}: The rounding mode could not be specified and the generated code may have different precision then the original code. Verify the correctness. SYCL math built-ins rounding mode is aligned with OpenCL C 1.2 standard.
  // CHECK-NEXT: */
  // CHECK: f2 = f0 * f1;
  f2 = __fmul_rn(f0, f1);
  // CHECK: /*
  // CHECK-NEXT: DPCT1013:{{[0-9]+}}: The rounding mode could not be specified and the generated code may have different precision then the original code. Verify the correctness. SYCL math built-ins rounding mode is aligned with OpenCL C 1.2 standard.
  // CHECK-NEXT: */
  // CHECK: f2 = f0 * f1;
  f2 = __fmul_ru(f0, f1);
  // CHECK: /*
  // CHECK-NEXT: DPCT1013:{{[0-9]+}}: The rounding mode could not be specified and the generated code may have different precision then the original code. Verify the correctness. SYCL math built-ins rounding mode is aligned with OpenCL C 1.2 standard.
  // CHECK-NEXT: */
  // CHECK: f2 = f0 * f1;
  f2 = __fmul_rz(f0, f1);

  // CHECK: /*
  // CHECK-NEXT: DPCT1013:{{[0-9]+}}: The rounding mode could not be specified and the generated code may have different precision then the original code. Verify the correctness. SYCL math built-ins rounding mode is aligned with OpenCL C 1.2 standard.
  // CHECK-NEXT: */
  // CHECK-NEXT: f1 = sycl::native::recip(f0);
  f1 = __frcp_rd(f0);
  // CHECK: /*
  // CHECK-NEXT: DPCT1013:{{[0-9]+}}: The rounding mode could not be specified and the generated code may have different precision then the original code. Verify the correctness. SYCL math built-ins rounding mode is aligned with OpenCL C 1.2 standard.
  // CHECK-NEXT: */
  // CHECK-NEXT: f1 = sycl::native::recip(f0);
  f1 = __frcp_rn(f0);
  // CHECK: /*
  // CHECK-NEXT: DPCT1013:{{[0-9]+}}: The rounding mode could not be specified and the generated code may have different precision then the original code. Verify the correctness. SYCL math built-ins rounding mode is aligned with OpenCL C 1.2 standard.
  // CHECK-NEXT: */
  // CHECK-NEXT: f1 = sycl::native::recip(f0);
  f1 = __frcp_ru(f0);
  // CHECK: /*
  // CHECK-NEXT: DPCT1013:{{[0-9]+}}: The rounding mode could not be specified and the generated code may have different precision then the original code. Verify the correctness. SYCL math built-ins rounding mode is aligned with OpenCL C 1.2 standard.
  // CHECK-NEXT: */
  // CHECK-NEXT: f1 = sycl::native::recip(f0);
  f1 = __frcp_rz(f0);

  // CHECK: /*
  // CHECK-NEXT: DPCT1013:{{[0-9]+}}: The rounding mode could not be specified and the generated code may have different precision then the original code. Verify the correctness. SYCL math built-ins rounding mode is aligned with OpenCL C 1.2 standard.
  // CHECK-NEXT: */
  // CHECK-NEXT: f1 = sycl::native::recip((float)i);
  f1 = __frcp_rd(i);
  // CHECK: /*
  // CHECK-NEXT: DPCT1013:{{[0-9]+}}: The rounding mode could not be specified and the generated code may have different precision then the original code. Verify the correctness. SYCL math built-ins rounding mode is aligned with OpenCL C 1.2 standard.
  // CHECK-NEXT: */
  // CHECK-NEXT: f1 = sycl::native::recip((float)i);
  f1 = __frcp_rn(i);
  // CHECK: /*
  // CHECK-NEXT: DPCT1013:{{[0-9]+}}: The rounding mode could not be specified and the generated code may have different precision then the original code. Verify the correctness. SYCL math built-ins rounding mode is aligned with OpenCL C 1.2 standard.
  // CHECK-NEXT: */
  // CHECK-NEXT: f1 = sycl::native::recip((float)i);
  f1 = __frcp_ru(i);
  // CHECK: /*
  // CHECK-NEXT: DPCT1013:{{[0-9]+}}: The rounding mode could not be specified and the generated code may have different precision then the original code. Verify the correctness. SYCL math built-ins rounding mode is aligned with OpenCL C 1.2 standard.
  // CHECK-NEXT: */
  // CHECK-NEXT: f1 = sycl::native::recip((float)i);
  f1 = __frcp_rz(i);

  // CHECK: /*
  // CHECK-NEXT: DPCT1013:{{[0-9]+}}: The rounding mode could not be specified and the generated code may have different precision then the original code. Verify the correctness. SYCL math built-ins rounding mode is aligned with OpenCL C 1.2 standard.
  // CHECK-NEXT: */
  // CHECK-NEXT: f0 = sycl::sqrt(f0);
  f0 = __fsqrt_rd(f0);
  // CHECK: /*
  // CHECK-NEXT: DPCT1013:{{[0-9]+}}: The rounding mode could not be specified and the generated code may have different precision then the original code. Verify the correctness. SYCL math built-ins rounding mode is aligned with OpenCL C 1.2 standard.
  // CHECK-NEXT: */
  // CHECK-NEXT: f1 = sycl::sqrt(f1);
  f1 = __fsqrt_rn(f1);
  // CHECK: /*
  // CHECK-NEXT: DPCT1013:{{[0-9]+}}: The rounding mode could not be specified and the generated code may have different precision then the original code. Verify the correctness. SYCL math built-ins rounding mode is aligned with OpenCL C 1.2 standard.
  // CHECK-NEXT: */
  // CHECK-NEXT: f0 = sycl::sqrt(f0);
  f0 = __fsqrt_ru(f0);
  // CHECK: /*
  // CHECK-NEXT: DPCT1013:{{[0-9]+}}: The rounding mode could not be specified and the generated code may have different precision then the original code. Verify the correctness. SYCL math built-ins rounding mode is aligned with OpenCL C 1.2 standard.
  // CHECK-NEXT: */
  // CHECK-NEXT: f1 = sycl::sqrt(f1);
  f1 = __fsqrt_rz(f1);

  // CHECK: /*
  // CHECK-NEXT: DPCT1013:{{[0-9]+}}: The rounding mode could not be specified and the generated code may have different precision then the original code. Verify the correctness. SYCL math built-ins rounding mode is aligned with OpenCL C 1.2 standard.
  // CHECK-NEXT: */
  // CHECK-NEXT: f0 = sycl::sqrt((float)i);
  f0 = __fsqrt_rd(i);
  // CHECK: /*
  // CHECK-NEXT: DPCT1013:{{[0-9]+}}: The rounding mode could not be specified and the generated code may have different precision then the original code. Verify the correctness. SYCL math built-ins rounding mode is aligned with OpenCL C 1.2 standard.
  // CHECK-NEXT: */
  // CHECK-NEXT: f1 = sycl::sqrt((float)i);
  f1 = __fsqrt_rn(i);
  // CHECK: /*
  // CHECK-NEXT: DPCT1013:{{[0-9]+}}: The rounding mode could not be specified and the generated code may have different precision then the original code. Verify the correctness. SYCL math built-ins rounding mode is aligned with OpenCL C 1.2 standard.
  // CHECK-NEXT: */
  // CHECK-NEXT: f0 = sycl::sqrt((float)i);
  f0 = __fsqrt_ru(i);
  // CHECK: /*
  // CHECK-NEXT: DPCT1013:{{[0-9]+}}: The rounding mode could not be specified and the generated code may have different precision then the original code. Verify the correctness. SYCL math built-ins rounding mode is aligned with OpenCL C 1.2 standard.
  // CHECK-NEXT: */
  // CHECK-NEXT: f1 = sycl::sqrt((float)i);
  f1 = __fsqrt_rz(i);

  // CHECK: /*
  // CHECK-NEXT: DPCT1013:{{[0-9]+}}: The rounding mode could not be specified and the generated code may have different precision then the original code. Verify the correctness. SYCL math built-ins rounding mode is aligned with OpenCL C 1.2 standard.
  // CHECK-NEXT: */
  // CHECK-NEXT: f2 = f0 - f1;
  f2 = __fsub_rd(f0, f1);
  // CHECK: /*
  // CHECK-NEXT: DPCT1013:{{[0-9]+}}: The rounding mode could not be specified and the generated code may have different precision then the original code. Verify the correctness. SYCL math built-ins rounding mode is aligned with OpenCL C 1.2 standard.
  // CHECK-NEXT: */
  // CHECK-NEXT: f2 = f0 - f1;
  f2 = __fsub_rn(f0, f1);
  // CHECK: /*
  // CHECK-NEXT: DPCT1013:{{[0-9]+}}: The rounding mode could not be specified and the generated code may have different precision then the original code. Verify the correctness. SYCL math built-ins rounding mode is aligned with OpenCL C 1.2 standard.
  // CHECK-NEXT: */
  // CHECK-NEXT: f2 = f0 - f1;
  f2 = __fsub_ru(f0, f1);
  // CHECK: /*
  // CHECK-NEXT: DPCT1013:{{[0-9]+}}: The rounding mode could not be specified and the generated code may have different precision then the original code. Verify the correctness. SYCL math built-ins rounding mode is aligned with OpenCL C 1.2 standard.
  // CHECK-NEXT: */
  // CHECK-NEXT: f2 = f0 - f1;
  f2 = __fsub_rz(f0, f1);

  // CHECK: f1 = sycl::log10(f1);
  f1 = __log10f(f1);
  // CHECK: f1 = sycl::log10((float)i);
  f1 = __log10f(i);

  // CHECK: f1 = sycl::log2(f1);
  f1 = __log2f(f1);
  // CHECK: f1 = sycl::log2((float)i);
  f1 = __log2f(i);

  // CHECK: f1 = sycl::log(f1);
  f1 = __logf(f1);
  // CHECK: f1 = sycl::log((float)i);
  f1 = __logf(i);

  // CHECK: f2 = sycl::pow(f0, f1);
  f2 = __powf(f0, f1);
  // CHECK: f2 = sycl::pow((float)i, (float)i);
  f2 = __powf(i, i);
  // CHECK: f2 = sycl::pow(f0, (float)i);
  f2 = __powf(f0, i);
  // CHECK: f2 = sycl::pow((float)i, f1);
  f2 = __powf(i, f1);

  // CHECK: f1 = sycl::sincos(f0, sycl::make_ptr<float, sycl::access::address_space::global_space>(&f2));
  __sincosf(f0, &f1, &f2);
  // CHECK: f1 = sycl::sincos((float)i, sycl::make_ptr<float, sycl::access::address_space::global_space>(&f2));
  __sincosf(i, &f1, &f2);

  // CHECK: f1 = sycl::sin(f1);
  f1 = __sinf(f1);
  // CHECK: f1 = sycl::sin((float)i);
  f1 = __sinf(i);

  // CHECK: f1 = sycl::tan(f1);
  f1 = __tanf(f1);
  // CHECK: f1 = sycl::tan((float)i);
  f1 = __tanf(i);

  // CHECK: f0 = sycl::fmin(f0, f1);
  f0 = fminf(f0, f1);
  // CHECK: f0 = sycl::fmin((float)i, (float)i);
  f0 = fminf(i, i);
  // CHECK: f0 = sycl::fmin(f0, (float)i);
  f0 = fminf(f0, i);
  // CHECK: f0 = sycl::fmin((float)i, f1);
  f0 = fminf(i, f1);

  // CHECK: f2 = sycl::fmax(f0, f1);
  f2 = fmaxf(f0, f1);
  // CHECK: f2 = sycl::fmax((float)i, (float)i);
  f2 = fmaxf(i, i);
  // CHECK: f2 = sycl::fmax(f0, (float)i);
  f2 = fmaxf(f0, i);
  // CHECK: f2 = sycl::fmax((float)i, f1);
  f2 = fmaxf(i, f1);

  // CHECK: f1 = sycl::floor(f1);
  f1 = floorf(f1);
  // CHECK: f1 = sycl::floor((float)i);
  f1 = floorf(i);

  // CHECK: f2 = sycl::ceil(f2);
  f2 = ceilf(f2);
  // CHECK: f2 = sycl::ceil((float)i);
  f2 = ceilf(i);

  // CHECK: f2 = sycl::fma(f0, f1, f2);
  f2 = fmaf(f0, f1, f2);
  // CHECK: f2 = sycl::fma((float)i, (float)i, (float)i);
  f2 = fmaf(i, i, i);
  // CHECK: f2 = sycl::fma(f0, (float)i, (float)i);
  f2 = fmaf(f0, i, i);
  // CHECK: f2 = sycl::fma((float)i, f1, (float)i);
  f2 = fmaf(i, f1, i);
  // CHECK: f2 = sycl::fma((float)i, (float)i, f2);
  f2 = fmaf(i, i, f2);
  // CHECK: f2 = sycl::fma(f0, f1, (float)i);
  f2 = fmaf(f0, f1, i);
  // CHECK: f2 = sycl::fma(f0, (float)i, f2);
  f2 = fmaf(f0, i, f2);
  // CHECK: f2 = sycl::fma((float)i, f1, f2);
  f2 = fmaf(i, f1, f2);

  // CHECK: f2 = sycl::nan(0u);
  f2 = nanf("NaN");

  // CHECK: /*
  // CHECK-NEXT: DPCT1013:{{[0-9]+}}: The rounding mode could not be specified and the generated code may have different precision then the original code. Verify the correctness. SYCL math built-ins rounding mode is aligned with OpenCL C 1.2 standard.
  // CHECK-NEXT: */
  // CHECK-NEXT: f2 = sycl::rsqrt(f2);
  f2 = __frsqrt_rn(f2);

  // CHECK: /*
  // CHECK-NEXT: DPCT1013:{{[0-9]+}}: The rounding mode could not be specified and the generated code may have different precision then the original code. Verify the correctness. SYCL math built-ins rounding mode is aligned with OpenCL C 1.2 standard.
  // CHECK-NEXT: */
  // CHECK-NEXT: f2 = sycl::rsqrt((float)i);
  f2 = __frsqrt_rn(i);

  // CHECK: f0 = sycl::nextafter(f0, f0);
  f0 = nextafterf(f0, f0);
  // CHECK: f0 = sycl::nextafter((float)i, (float)i);
  f0 = nextafterf(i, i);
  // CHECK: f0 = sycl::nextafter(f0, (float)i);
  f0 = nextafterf(f0, i);
  // CHECK: f0 = sycl::nextafter((float)i, f1);
  f0 = nextafterf(i, f1);
}

__global__ void kernelFuncTypecasts() {
  short s, s_1;
  unsigned short us;
  int i, i_1;
  unsigned int ui, ui_1;
  long l;
  unsigned long ul;
  long long ll;
  unsigned long long ull;

  __half h;
  __half2 h2;
  float f;
  float2 f2;
  double d;
  double2 d2;

  // CHECK: h2 = f2.convert<sycl::half, sycl::rounding_mode::rte>();
  h2 = __float22half2_rn(f2);

  // CHECK: h = sycl::vec<float, 1>{f}.convert<sycl::half, sycl::rounding_mode::automatic>().get_value(0);
  h = __float2half(f);

  // CHECK: h2 = sycl::float2{f,f}.convert<sycl::half, sycl::rounding_mode::rte>();
  h2 = __float2half2_rn(f);

  // CHECK: h = sycl::vec<float, 1>{f}.convert<sycl::half, sycl::rounding_mode::rtn>().get_value(0);
  h = __float2half_rd(f);

  // sycl::vec<float, 1>{f}.convert<sycl::half, sycl::rounding_mode::rte>().get_value(0);
  __float2half_rn(f);

  // CHECK: h = sycl::vec<float, 1>{f}.convert<sycl::half, sycl::rounding_mode::rtp>().get_value(0);
  h = __float2half_ru(f);

  // CHECK: h = sycl::vec<float, 1>{f}.convert<sycl::half, sycl::rounding_mode::rtz>().get_value(0);
  h = __float2half_rz(f);

  // CHECK: h2 = sycl::float2{f,f}.convert<sycl::half, sycl::rounding_mode::rte>();
  h2 = __floats2half2_rn(f, f);

  // CHECK: f2 = h2.convert<float, sycl::rounding_mode::automatic>();
  f2 = __half22float2(h2);

  // CHECK: f = sycl::vec<sycl::half, 1>{h}.convert<float, sycl::rounding_mode::automatic>().get_value(0);
  f = __half2float(h);

  // CHECK: h2 = sycl::half2{h,h};
  h2 = __half2half2(h);

  // CHECK: i = sycl::vec<sycl::half, 1>{h}.convert<int, sycl::rounding_mode::rtn>().get_value(0);
  i = __half2int_rd(h);

  // CHECK: i = sycl::vec<sycl::half, 1>{h}.convert<int, sycl::rounding_mode::rte>().get_value(0);
  i = __half2int_rn(h);

  // CHECK: i = sycl::vec<sycl::half, 1>{h}.convert<int, sycl::rounding_mode::rtp>().get_value(0);
  i = __half2int_ru(h);

  // CHECK: i = sycl::vec<sycl::half, 1>{h}.convert<int, sycl::rounding_mode::rtz>().get_value(0);
  i = __half2int_rz(h);

  // CHECK: ll = sycl::vec<sycl::half, 1>{h}.convert<long long, sycl::rounding_mode::rtn>().get_value(0);
  ll = __half2ll_rd(h);

  // CHECK: ll = sycl::vec<sycl::half, 1>{h}.convert<long long, sycl::rounding_mode::rte>().get_value(0);
  ll = __half2ll_rn(h);

  // CHECK: ll = sycl::vec<sycl::half, 1>{h}.convert<long long, sycl::rounding_mode::rtp>().get_value(0);
  ll = __half2ll_ru(h);

  // CHECK: ll = sycl::vec<sycl::half, 1>{h}.convert<long long, sycl::rounding_mode::rtz>().get_value(0);
  ll = __half2ll_rz(h);

  // CHECK: s = sycl::vec<sycl::half, 1>{h}.convert<short, sycl::rounding_mode::rtn>().get_value(0);
  s = __half2short_rd(h);

  // CHECK: s = sycl::vec<sycl::half, 1>{h}.convert<short, sycl::rounding_mode::rte>().get_value(0);
  s = __half2short_rn(h);

  // CHECK: s = sycl::vec<sycl::half, 1>{h}.convert<short, sycl::rounding_mode::rtp>().get_value(0);
  s = __half2short_ru(h);

  // CHECK: s = sycl::vec<sycl::half, 1>{h}.convert<short, sycl::rounding_mode::rtz>().get_value(0);
  s = __half2short_rz(h);

  // CHECK: ui = sycl::vec<sycl::half, 1>{h}.convert<unsigned int, sycl::rounding_mode::rtn>().get_value(0);
  ui = __half2uint_rd(h);

  // CHECK: ui = sycl::vec<sycl::half, 1>{h}.convert<unsigned int, sycl::rounding_mode::rte>().get_value(0);
  ui = __half2uint_rn(h);

  // CHECK:ui = sycl::vec<sycl::half, 1>{h}.convert<unsigned int, sycl::rounding_mode::rtp>().get_value(0);
  ui = __half2uint_ru(h);

  // CHECK: ui = sycl::vec<sycl::half, 1>{h}.convert<unsigned int, sycl::rounding_mode::rtz>().get_value(0);
  ui = __half2uint_rz(h);

  // CHECK: ull = sycl::vec<sycl::half, 1>{h}.convert<unsigned long long, sycl::rounding_mode::rtn>().get_value(0);
  ull = __half2ull_rd(h);

  // CHECK: ull = sycl::vec<sycl::half, 1>{h}.convert<unsigned long long, sycl::rounding_mode::rte>().get_value(0);
  ull = __half2ull_rn(h);

  // CHECK: ull = sycl::vec<sycl::half, 1>{h}.convert<unsigned long long, sycl::rounding_mode::rtp>().get_value(0);
  ull = __half2ull_ru(h);

  // CHECK: ull = sycl::vec<sycl::half, 1>{h}.convert<unsigned long long, sycl::rounding_mode::rtz>().get_value(0);
  ull = __half2ull_rz(h);

  // CHECK: us = sycl::vec<sycl::half, 1>{h}.convert<unsigned short, sycl::rounding_mode::rtn>().get_value(0);
  us = __half2ushort_rd(h);

  // CHECK: us = sycl::vec<sycl::half, 1>{h}.convert<unsigned short, sycl::rounding_mode::rte>().get_value(0);
  us = __half2ushort_rn(h);

  // CHECK: us = sycl::vec<sycl::half, 1>{h}.convert<unsigned short, sycl::rounding_mode::rtp>().get_value(0);
  us = __half2ushort_ru(h);

  // CHECK: us = sycl::vec<sycl::half, 1>{h}.convert<unsigned short, sycl::rounding_mode::rtz>().get_value(0);
  us = __half2ushort_rz(h);

  // CHECK: s = dpct::bit_cast<sycl::half, short>(h);
  s = __half_as_short(h);

  // CHECK: us = dpct::bit_cast<sycl::half, unsigned short>(h);
  us = __half_as_ushort(h);

  // CHECK: h2 = sycl::half2{h,h};
  h2 = __halves2half2(h, h);

  // CHECK: f = h2.get_value(0);
  f = __high2float(h2);

  // CHECK: h = h2.get_value(0);
  h = __high2half(h2);

  // CHECK: h2 = sycl::half2{h2.get_value(0), h2.get_value(0)};
  h2 = __high2half2(h2);

  // CHECK: h2 = sycl::half2{h2.get_value(0), h2.get_value(0)};
  h2 = __highs2half2(h2, h2);

  // CHECK: h = sycl::vec<int, 1>{i}.convert<sycl::half, sycl::rounding_mode::rtn>().get_value(0);
  h = __int2half_rd(i);

  // CHECK: h = sycl::vec<int, 1>{i}.convert<sycl::half, sycl::rounding_mode::rte>().get_value(0);
  h = __int2half_rn(i);

  // CHECK: h = sycl::vec<int, 1>{i}.convert<sycl::half, sycl::rounding_mode::rtp>().get_value(0);
  h = __int2half_ru(i);

  // CHECK: h = sycl::vec<int, 1>{i}.convert<sycl::half, sycl::rounding_mode::rtz>().get_value(0);
  h = __int2half_rz(i);

  // CHECK: h = sycl::vec<long long, 1>{ll}.convert<sycl::half, sycl::rounding_mode::rtn>().get_value(0);
  h = __ll2half_rd(ll);

  // CHECK: h = sycl::vec<long long, 1>{ll}.convert<sycl::half, sycl::rounding_mode::rte>().get_value(0);
  h = __ll2half_rn(ll);

  // CHECK: h = sycl::vec<long long, 1>{ll}.convert<sycl::half, sycl::rounding_mode::rtp>().get_value(0);
  h = __ll2half_ru(ll);

  // CHECK: h = sycl::vec<long long, 1>{ll}.convert<sycl::half, sycl::rounding_mode::rtz>().get_value(0);
  h = __ll2half_rz(ll);

  // CHECK: f = h2.get_value(1);
  f = __low2float(h2);

  // CHECK: h = h2.get_value(1);
  h = __low2half(h2);

  // CHECK: h2 = sycl::half2{h2.get_value(1), h2.get_value(1)};
  h2 = __low2half2(h2);

  // CHECK: h2 = sycl::half2{h2.get_value(1), h2.get_value(0)};
  h2 = __lowhigh2highlow(h2);

  // CHECK: h2 = sycl::half2{h2.get_value(1), h2.get_value(1)};
  h2 = __lows2half2(h2, h2);

  // CHECK: h = sycl::vec<short, 1>{s}.convert<sycl::half, sycl::rounding_mode::rtn>().get_value(0);
  h = __short2half_rd(s);

  // CHECK: h = sycl::vec<short, 1>{s}.convert<sycl::half, sycl::rounding_mode::rte>().get_value(0);
  h = __short2half_rn(s);

  // CHECK: h = sycl::vec<short, 1>{s}.convert<sycl::half, sycl::rounding_mode::rtp>().get_value(0);
  h = __short2half_ru(s);

  // CHECK: h = sycl::vec<short, 1>{s}.convert<sycl::half, sycl::rounding_mode::rtz>().get_value(0);
  h = __short2half_rz(s);

  // CHECK: h = dpct::bit_cast<short, sycl::half>(s);
  h = __short_as_half(s);

  // CHECK: h = sycl::vec<unsigned int, 1>{ui}.convert<sycl::half, sycl::rounding_mode::rtn>().get_value(0);
  h = __uint2half_rd(ui);

  // CHECK: h = sycl::vec<unsigned int, 1>{ui}.convert<sycl::half, sycl::rounding_mode::rte>().get_value(0);
  h = __uint2half_rn(ui);

  // CHECK: h = sycl::vec<unsigned int, 1>{ui}.convert<sycl::half, sycl::rounding_mode::rtp>().get_value(0);
  h = __uint2half_ru(ui);

  // CHECK: h = sycl::vec<unsigned int, 1>{ui}.convert<sycl::half, sycl::rounding_mode::rtz>().get_value(0);
  h = __uint2half_rz(ui);

  // CHECK: h = sycl::vec<unsigned long long, 1>{ull}.convert<sycl::half, sycl::rounding_mode::rtn>().get_value(0);
  h = __ull2half_rd(ull);

  // CHECK: h = sycl::vec<unsigned long long, 1>{ull}.convert<sycl::half, sycl::rounding_mode::rte>().get_value(0);
  h = __ull2half_rn(ull);

  // CHECK: h = sycl::vec<unsigned long long, 1>{ull}.convert<sycl::half, sycl::rounding_mode::rtp>().get_value(0);
  h = __ull2half_ru(ull);

  // CHECK: h = sycl::vec<unsigned long long, 1>{ull}.convert<sycl::half, sycl::rounding_mode::rtz>().get_value(0);
  h = __ull2half_rz(ull);

  // CHECK: h = sycl::vec<unsigned short, 1>{us}.convert<sycl::half, sycl::rounding_mode::rtn>().get_value(0);
  h = __ushort2half_rd(us);

  // CHECK: h = sycl::vec<unsigned short, 1>{us}.convert<sycl::half, sycl::rounding_mode::rte>().get_value(0);
  h = __ushort2half_rn(us);

  // CHECK: h = sycl::vec<unsigned short, 1>{us}.convert<sycl::half, sycl::rounding_mode::rtp>().get_value(0);
  h = __ushort2half_ru(us);

  // CHECK: h = sycl::vec<unsigned short, 1>{us}.convert<sycl::half, sycl::rounding_mode::rtz>().get_value(0);
  h = __ushort2half_rz(us);

  // CHECK: h = dpct::bit_cast<unsigned short, sycl::half>(us);
  h = __ushort_as_half(us);

  // CHECK: f = sycl::vec<double, 1>{d}.convert<float, sycl::rounding_mode::rtn>().get_value(0);
  f = __double2float_rd(d);

  // CHECK: f = sycl::vec<double, 1>{d}.convert<float, sycl::rounding_mode::rte>().get_value(0);
  f = __double2float_rn(d);

  // CHECK: f = sycl::vec<double, 1>{d}.convert<float, sycl::rounding_mode::rtp>().get_value(0);
  f = __double2float_ru(d);

  // CHECK: f = sycl::vec<double, 1>{d}.convert<float, sycl::rounding_mode::rtz>().get_value(0);
  f = __double2float_rz(d);

  // CHECK: i = sycl::vec<double, 1>{d}.convert<int, sycl::rounding_mode::rtn>().get_value(0);
  i = __double2int_rd(d);

  // CHECK: i = sycl::vec<double, 1>{d}.convert<int, sycl::rounding_mode::rte>().get_value(0);
  i = __double2int_rn(d);

  // CHECK: i = sycl::vec<double, 1>{d}.convert<int, sycl::rounding_mode::rtp>().get_value(0);
  i = __double2int_ru(d);

  // CHECK: i = sycl::vec<double, 1>{d}.convert<int, sycl::rounding_mode::rtz>().get_value(0);
  i = __double2int_rz(d);

  // CHECK: ll = sycl::vec<double, 1>{d}.convert<long long, sycl::rounding_mode::rtn>().get_value(0);
  ll = __double2ll_rd(d);

  // CHECK: ll = sycl::vec<double, 1>{d}.convert<long long, sycl::rounding_mode::rte>().get_value(0);
  ll = __double2ll_rn(d);

  // CHECK: ll = sycl::vec<double, 1>{d}.convert<long long, sycl::rounding_mode::rtp>().get_value(0);
  ll = __double2ll_ru(d);

  // CHECK: ll = sycl::vec<double, 1>{d}.convert<long long, sycl::rounding_mode::rtz>().get_value(0);
  ll = __double2ll_rz(d);

  // CHECK: ui = sycl::vec<double, 1>{d}.convert<unsigned int, sycl::rounding_mode::rtn>().get_value(0);
  ui = __double2uint_rd(d);

  // CHECK:ui = sycl::vec<double, 1>{d}.convert<unsigned int, sycl::rounding_mode::rte>().get_value(0);
  ui = __double2uint_rn(d);

  // CHECK: ui = sycl::vec<double, 1>{d}.convert<unsigned int, sycl::rounding_mode::rtp>().get_value(0);
  ui = __double2uint_ru(d);

  // CHECK: ui = sycl::vec<double, 1>{d}.convert<unsigned int, sycl::rounding_mode::rtz>().get_value(0);
  ui = __double2uint_rz(d);

  // CHECK: ull = sycl::vec<double, 1>{d}.convert<unsigned long long, sycl::rounding_mode::rtn>().get_value(0);
  ull = __double2ull_rd(d);

  // CHECK: ull = sycl::vec<double, 1>{d}.convert<unsigned long long, sycl::rounding_mode::rte>().get_value(0);
  ull = __double2ull_rn(d);

  // CHECK: ull = sycl::vec<double, 1>{d}.convert<unsigned long long, sycl::rounding_mode::rtp>().get_value(0);
  ull = __double2ull_ru(d);

  // CHECK: ull = sycl::vec<double, 1>{d}.convert<unsigned long long, sycl::rounding_mode::rtz>().get_value(0);
  ull = __double2ull_rz(d);

  // CHECK: ll = dpct::bit_cast<double, long long>(d);
  ll = __double_as_longlong(d);

  // CHECK: i = sycl::vec<float, 1>{f}.convert<int, sycl::rounding_mode::rtn>().get_value(0);
  i = __float2int_rd(f);

  // CHECK: i = sycl::vec<float, 1>{f}.convert<int, sycl::rounding_mode::rte>().get_value(0);
  i = __float2int_rn(f);

  // CHECK: i = sycl::vec<float, 1>{f}.convert<int, sycl::rounding_mode::rtp>().get_value(0);
  i = __float2int_ru(f);

  // CHECK: i = sycl::vec<float, 1>{f}.convert<int, sycl::rounding_mode::rtz>().get_value(0);
  i = __float2int_rz(f);

  // CHECK: ll = sycl::vec<float, 1>{f}.convert<long long, sycl::rounding_mode::rtn>().get_value(0);
  ll = __float2ll_rd(f);

  // CHECK: ll = sycl::vec<float, 1>{f}.convert<long long, sycl::rounding_mode::rte>().get_value(0);
  ll = __float2ll_rn(f);

  // CHECK: ll = sycl::vec<float, 1>{f}.convert<long long, sycl::rounding_mode::rtp>().get_value(0);
  ll = __float2ll_ru(f);

  // CHECK: ll = sycl::vec<float, 1>{f}.convert<long long, sycl::rounding_mode::rtz>().get_value(0);
  ll = __float2ll_rz(f);

  // CHECK: ui = sycl::vec<float, 1>{f}.convert<unsigned int, sycl::rounding_mode::rtn>().get_value(0);
  ui = __float2uint_rd(f);

  // CHECK: ui = sycl::vec<float, 1>{f}.convert<unsigned int, sycl::rounding_mode::rte>().get_value(0);
  ui = __float2uint_rn(f);

  // CHECK: ui = sycl::vec<float, 1>{f}.convert<unsigned int, sycl::rounding_mode::rtp>().get_value(0);
  ui = __float2uint_ru(f);

  // CHECK: ui = sycl::vec<float, 1>{f}.convert<unsigned int, sycl::rounding_mode::rtz>().get_value(0);
  ui = __float2uint_rz(f);

  // CHECK: ull = sycl::vec<float, 1>{f}.convert<unsigned long long, sycl::rounding_mode::rtn>().get_value(0);
  ull = __float2ull_rd(f);

  // CHECK: ull = sycl::vec<float, 1>{f}.convert<unsigned long long, sycl::rounding_mode::rte>().get_value(0);
  ull = __float2ull_rn(f);

  // CHECK: ull = sycl::vec<float, 1>{f}.convert<unsigned long long, sycl::rounding_mode::rtp>().get_value(0);
  ull = __float2ull_ru(f);

  // CHECK: ull = sycl::vec<float, 1>{f}.convert<unsigned long long, sycl::rounding_mode::rtz>().get_value(0);
  ull = __float2ull_rz(f);

  // CHECK: i = dpct::bit_cast<float, int>(f);
  i = __float_as_int(f);

  // CHECK: ui = dpct::bit_cast<float, unsigned int>(f);
  ui = __float_as_uint(f);

  // CHECK: d = sycl::vec<int, 1>{i}.convert<double, sycl::rounding_mode::rte>().get_value(0);
  d = __int2double_rn(i);

  // CHECK: d = sycl::vec<int, 1>{i}.convert<float, sycl::rounding_mode::rtn>().get_value(0);
  d = __int2float_rd(i);

  // CHECK: d = sycl::vec<int, 1>{i}.convert<float, sycl::rounding_mode::rte>().get_value(0);
  d = __int2float_rn(i);

  // CHECK: d = sycl::vec<int, 1>{i}.convert<float, sycl::rounding_mode::rtp>().get_value(0);
  d = __int2float_ru(i);

  // CHECK: d = sycl::vec<int, 1>{i}.convert<float, sycl::rounding_mode::rtz>().get_value(0);
  d = __int2float_rz(i);

  // CHECK: f = dpct::bit_cast<int, float>(i);
  f = __int_as_float(i);

  // CHECK: d = sycl::vec<long long, 1>{ll}.convert<double, sycl::rounding_mode::rtn>().get_value(0);
  d = __ll2double_rd(ll);

  // CHECK: d = sycl::vec<long long, 1>{ll}.convert<double, sycl::rounding_mode::rte>().get_value(0);
  d = __ll2double_rn(ll);

  // CHECK: d = sycl::vec<long long, 1>{ll}.convert<double, sycl::rounding_mode::rtp>().get_value(0);
  d = __ll2double_ru(ll);

  // CHECK: d = sycl::vec<long long, 1>{ll}.convert<double, sycl::rounding_mode::rtz>().get_value(0);
  d = __ll2double_rz(ll);

  // CHECK: f = sycl::vec<long long, 1>{ll}.convert<float, sycl::rounding_mode::rtn>().get_value(0);
  f = __ll2float_rd(ll);

  // CHECK: f = sycl::vec<long long, 1>{ll}.convert<float, sycl::rounding_mode::rte>().get_value(0);
  f = __ll2float_rn(ll);

  // CHECK: f = sycl::vec<long long, 1>{ll}.convert<float, sycl::rounding_mode::rtp>().get_value(0);
  f = __ll2float_ru(ll);

  // CHECK: f = sycl::vec<long long, 1>{ll}.convert<float, sycl::rounding_mode::rtz>().get_value(0);
  f = __ll2float_rz(ll);

  // CHECK: d = dpct::bit_cast<long long, double>(ll);
  d = __longlong_as_double(ll);

  // CHECK: d = sycl::vec<unsigned int, 1>{ui}.convert<double, sycl::rounding_mode::rte>().get_value(0);
  d = __uint2double_rn(ui);

  // CHECK: f = sycl::vec<unsigned int, 1>{ui}.convert<float, sycl::rounding_mode::rtn>().get_value(0);
  f = __uint2float_rd(ui);

  // CHECK: f = sycl::vec<unsigned int, 1>{ui}.convert<float, sycl::rounding_mode::rte>().get_value(0);
  f = __uint2float_rn(ui);

  // CHECK: f = sycl::vec<unsigned int, 1>{ui}.convert<float, sycl::rounding_mode::rtp>().get_value(0);
  f = __uint2float_ru(ui);

  // CHECK: f = sycl::vec<unsigned int, 1>{ui}.convert<float, sycl::rounding_mode::rtz>().get_value(0);
  f = __uint2float_rz(ui);

  // CHECK: f = dpct::bit_cast<unsigned int, float>(ui);
  f = __uint_as_float(ui);

  // CHECK: d = sycl::vec<unsigned long long, 1>{ull}.convert<double, sycl::rounding_mode::rtn>().get_value(0);
  d = __ull2double_rd(ull);

  // CHECK: d = sycl::vec<unsigned long long, 1>{ull}.convert<double, sycl::rounding_mode::rte>().get_value(0);
  d = __ull2double_rn(ull);

  // CHECK: d = sycl::vec<unsigned long long, 1>{ull}.convert<double, sycl::rounding_mode::rtp>().get_value(0);
  d = __ull2double_ru(ull);

  // CHECK: d = sycl::vec<unsigned long long, 1>{ull}.convert<double, sycl::rounding_mode::rtz>().get_value(0);
  d = __ull2double_rz(ull);

  // CHECK: f = sycl::vec<unsigned long long, 1>{ull}.convert<float, sycl::rounding_mode::rtn>().get_value(0);
  f = __ull2float_rd(ull);

  // CHECK: f = sycl::vec<unsigned long long, 1>{ull}.convert<float, sycl::rounding_mode::rte>().get_value(0);
  f = __ull2float_rn(ull);

  // CHECK: f = sycl::vec<unsigned long long, 1>{ull}.convert<float, sycl::rounding_mode::rtp>().get_value(0);
  f = __ull2float_ru(ull);

  // CHECK: f = sycl::vec<unsigned long long, 1>{ull}.convert<float, sycl::rounding_mode::rtz>().get_value(0);
  f = __ull2float_rz(ull);
}

void testDouble() {
  const unsigned int NUM = 3;
  const unsigned int bytes = NUM * sizeof(double);

  double *hostArrayDouble = (double *)malloc(bytes);
  memset(hostArrayDouble, 0, bytes);
  const long double pi = std::acos(-1.L);
  *hostArrayDouble = pi;
  *(hostArrayDouble + 1) = pi - 1;

  double *deviceArrayDouble;
  cudaMalloc((double **)&deviceArrayDouble, bytes);

  cudaMemcpy(deviceArrayDouble, hostArrayDouble, bytes, cudaMemcpyHostToDevice);

  kernelFuncDouble<<<1, 1>>>(deviceArrayDouble);

  cudaMemcpy(hostArrayDouble, deviceArrayDouble, bytes, cudaMemcpyDeviceToHost);

  cudaFree(deviceArrayDouble);

  cout << std::setprecision(std::numeric_limits<long double>::digits10 + 1)
       << *(hostArrayDouble + 2) << endl;
}

void testFloat() {
  const unsigned int NUM = 3;
  const unsigned int bytes = NUM * sizeof(float);

  float *hostArrayFloat = (float *)malloc(bytes);
  memset(hostArrayFloat, 0, bytes);
  const long double pi = std::acos(-1.L);
  *hostArrayFloat = pi;
  *(hostArrayFloat + 1) = pi - 1;

  float *deviceArrayFloat;
  cudaMalloc((float **)&deviceArrayFloat, bytes);

  cudaMemcpy(deviceArrayFloat, hostArrayFloat, bytes, cudaMemcpyHostToDevice);

  kernelFuncFloat<<<1, 1>>>(deviceArrayFloat);

  cudaMemcpy(hostArrayFloat, deviceArrayFloat, bytes, cudaMemcpyDeviceToHost);

  cudaFree(deviceArrayFloat);

  cout << std::setprecision(std::numeric_limits<long double>::digits10 + 1)
       << *(hostArrayFloat + 2) << endl;
}

__global__ void testUnsupported() {
  int i;
  unsigned u;
  long l;
  long long ll;
  unsigned long long ull;
  half h;
  float f;
  double d;
  half2 h2;
  bool b;

  // CHECK: /*
  // CHECK-NEXT: DPCT1004:{{[0-9]+}}: Could not generate replacement.
  // CHECK-NEXT: */
  __hadd_sat(h, h);
  // CHECK: /*
  // CHECK-NEXT: DPCT1004:{{[0-9]+}}: Could not generate replacement.
  // CHECK-NEXT: */
  __hfma_sat(h, h, h);
  // CHECK: /*
  // CHECK-NEXT: DPCT1004:{{[0-9]+}}: Could not generate replacement.
  // CHECK-NEXT: */
  __hmul_sat(h, h);
  // CHECK: /*
  // CHECK-NEXT: DPCT1004:{{[0-9]+}}: Could not generate replacement.
  // CHECK-NEXT: */
  __hsub_sat(h, h);

  // CHECK: /*
  // CHECK-NEXT: DPCT1004:{{[0-9]+}}: Could not generate replacement.
  // CHECK-NEXT: */
  __hadd2_sat(h2, h2);
  // CHECK: /*
  // CHECK-NEXT: DPCT1004:{{[0-9]+}}: Could not generate replacement.
  // CHECK-NEXT: */
  __hfma2_sat(h2, h2, h2);
  // CHECK: /*
  // CHECK-NEXT: DPCT1004:{{[0-9]+}}: Could not generate replacement.
  // CHECK-NEXT: */
  __hmul2_sat(h2, h2);
  // CHECK: /*
  // CHECK-NEXT: DPCT1004:{{[0-9]+}}: Could not generate replacement.
  // CHECK-NEXT: */
  __hsub2_sat(h2, h2);

  // CHECK: /*
  // CHECK-NEXT: DPCT1004:{{[0-9]+}}: Could not generate replacement.
  // CHECK-NEXT: */
  b = __hequ(h, h);
  // CHECK: /*
  // CHECK-NEXT: DPCT1004:{{[0-9]+}}: Could not generate replacement.
  // CHECK-NEXT: */
  b = __hgeu(h, h);
  // CHECK: /*
  // CHECK-NEXT: DPCT1004:{{[0-9]+}}: Could not generate replacement.
  // CHECK-NEXT: */
  b = __hgtu(h, h);
  // CHECK: /*
  // CHECK-NEXT: DPCT1004:{{[0-9]+}}: Could not generate replacement.
  // CHECK-NEXT: */
  b = __hleu(h, h);
  // CHECK: /*
  // CHECK-NEXT: DPCT1004:{{[0-9]+}}: Could not generate replacement.
  // CHECK-NEXT: */
  b = __hltu(h, h);
  // CHECK: /*
  // CHECK-NEXT: DPCT1004:{{[0-9]+}}: Could not generate replacement.
  // CHECK-NEXT: */
  b = __hneu(h, h);

  // CHECK: /*
  // CHECK-NEXT: DPCT1004:{{[0-9]+}}: Could not generate replacement.
  // CHECK-NEXT: */
  b = __hbeq2(h2, h2);
  // CHECK: /*
  // CHECK-NEXT: DPCT1004:{{[0-9]+}}: Could not generate replacement.
  // CHECK-NEXT: */
  b = __hbequ2(h2, h2);
  // CHECK: /*
  // CHECK-NEXT: DPCT1004:{{[0-9]+}}: Could not generate replacement.
  // CHECK-NEXT: */
  b = __hbge2(h2, h2);
  // CHECK: /*
  // CHECK-NEXT: DPCT1004:{{[0-9]+}}: Could not generate replacement.
  // CHECK-NEXT: */
  b = __hbgeu2(h2, h2);
  // CHECK: /*
  // CHECK-NEXT: DPCT1004:{{[0-9]+}}: Could not generate replacement.
  // CHECK-NEXT: */
  b = __hbgt2(h2, h2);
  // CHECK: /*
  // CHECK-NEXT: DPCT1004:{{[0-9]+}}: Could not generate replacement.
  // CHECK-NEXT: */
  b = __hbgtu2(h2, h2);
  // CHECK: /*
  // CHECK-NEXT: DPCT1004:{{[0-9]+}}: Could not generate replacement.
  // CHECK-NEXT: */
  b = __hble2(h2, h2);
  // CHECK: /*
  // CHECK-NEXT: DPCT1004:{{[0-9]+}}: Could not generate replacement.
  // CHECK-NEXT: */
  b = __hbleu2(h2, h2);
  // CHECK: /*
  // CHECK-NEXT: DPCT1004:{{[0-9]+}}: Could not generate replacement.
  // CHECK-NEXT: */
  b = __hblt2(h2, h2);
  // CHECK: /*
  // CHECK-NEXT: DPCT1004:{{[0-9]+}}: Could not generate replacement.
  // CHECK-NEXT: */
  b = __hbltu2(h2, h2);
  // CHECK: /*
  // CHECK-NEXT: DPCT1004:{{[0-9]+}}: Could not generate replacement.
  // CHECK-NEXT: */
  b = __hbne2(h2, h2);
  // CHECK: /*
  // CHECK-NEXT: DPCT1004:{{[0-9]+}}: Could not generate replacement.
  // CHECK-NEXT: */
  b = __hbneu2(h2, h2);

  // CHECK: /*
  // CHECK-NEXT: DPCT1004:{{[0-9]+}}: Could not generate replacement.
  // CHECK-NEXT: */
  __hequ2(h2, h2);
  // CHECK: /*
  // CHECK-NEXT: DPCT1004:{{[0-9]+}}: Could not generate replacement.
  // CHECK-NEXT: */
  __hgeu2(h2, h2);
  // CHECK: /*
  // CHECK-NEXT: DPCT1004:{{[0-9]+}}: Could not generate replacement.
  // CHECK-NEXT: */
  __hgtu2(h2, h2);
  // CHECK: /*
  // CHECK-NEXT: DPCT1004:{{[0-9]+}}: Could not generate replacement.
  // CHECK-NEXT: */
  __hleu2(h2, h2);
  // CHECK: /*
  // CHECK-NEXT: DPCT1004:{{[0-9]+}}: Could not generate replacement.
  // CHECK-NEXT: */
  __hltu2(h2, h2);
  // CHECK: /*
  // CHECK-NEXT: DPCT1004:{{[0-9]+}}: Could not generate replacement.
  // CHECK-NEXT: */
  __hneu2(h2, h2);

  // CHECK: /*
  // CHECK-NEXT: DPCT1004:{{[0-9]+}}: Could not generate replacement.
  // CHECK-NEXT: */
  f = cyl_bessel_i0f(f);
  // CHECK: /*
  // CHECK-NEXT: DPCT1004:{{[0-9]+}}: Could not generate replacement.
  // CHECK-NEXT: */
  f = cyl_bessel_i1f(f);
  // CHECK: /*
  // CHECK-NEXT: DPCT1004:{{[0-9]+}}: Could not generate replacement.
  // CHECK-NEXT: */
  f = erfcinvf(f);
  // CHECK: /*
  // CHECK-NEXT: DPCT1004:{{[0-9]+}}: Could not generate replacement.
  // CHECK-NEXT: */
  f = erfcxf(f);
  // CHECK: /*
  // CHECK-NEXT: DPCT1004:{{[0-9]+}}: Could not generate replacement.
  // CHECK-NEXT: */
  f = erfinvf(f);
  // CHECK: /*
  // CHECK-NEXT: DPCT1004:{{[0-9]+}}: Could not generate replacement.
  // CHECK-NEXT: */
  f = j0f(f);
  // CHECK: /*
  // CHECK-NEXT: DPCT1004:{{[0-9]+}}: Could not generate replacement.
  // CHECK-NEXT: */
  f = j1f(f);
  // CHECK: /*
  // CHECK-NEXT: DPCT1004:{{[0-9]+}}: Could not generate replacement.
  // CHECK-NEXT: */
  f = jnf(i, f);

  // CHECK: /*
  // CHECK-NEXT: DPCT1004:{{[0-9]+}}: Could not generate replacement.
  // CHECK-NEXT: */
  f = norm3df(f, f, f);
  // CHECK: /*
  // CHECK-NEXT: DPCT1004:{{[0-9]+}}: Could not generate replacement.
  // CHECK-NEXT: */
  f = norm4df(f, f, f, f);
  // CHECK: /*
  // CHECK-NEXT: DPCT1004:{{[0-9]+}}: Could not generate replacement.
  // CHECK-NEXT: */
  f = normcdff(f);
  // CHECK: /*
  // CHECK-NEXT: DPCT1004:{{[0-9]+}}: Could not generate replacement.
  // CHECK-NEXT: */
  f = normcdfinvf(f);
  // CHECK: /*
  // CHECK-NEXT: DPCT1004:{{[0-9]+}}: Could not generate replacement.
  // CHECK-NEXT: */
  f = normf(i, &f);
  // CHECK: /*
  // CHECK-NEXT: DPCT1004:{{[0-9]+}}: Could not generate replacement.
  // CHECK-NEXT: */
  f = rcbrtf(f);
  // CHECK: /*
  // CHECK-NEXT: DPCT1004:{{[0-9]+}}: Could not generate replacement.
  // CHECK-NEXT: */
  f = rnorm3df(f, f, f);
  // CHECK: /*
  // CHECK-NEXT: DPCT1004:{{[0-9]+}}: Could not generate replacement.
  // CHECK-NEXT: */
  f = rnorm4df(f, f, f, f);
  // CHECK: /*
  // CHECK-NEXT: DPCT1004:{{[0-9]+}}: Could not generate replacement.
  // CHECK-NEXT: */
  f = rnormf(i, &f);
  // CHECK: /*
  // CHECK-NEXT: DPCT1004:{{[0-9]+}}: Could not generate replacement.
  // CHECK-NEXT: */
  f = scalblnf(f, l);
  // CHECK: /*
  // CHECK-NEXT: DPCT1004:{{[0-9]+}}: Could not generate replacement.
  // CHECK-NEXT: */
  f = scalbnf(f, i);
  // CHECK: /*
  // CHECK-NEXT: DPCT1004:{{[0-9]+}}: Could not generate replacement.
  // CHECK-NEXT: */
  f = y0f(f);
  // CHECK: /*
  // CHECK-NEXT: DPCT1004:{{[0-9]+}}: Could not generate replacement.
  // CHECK-NEXT: */
  f = y1f(f);
  // CHECK: /*
  // CHECK-NEXT: DPCT1004:{{[0-9]+}}: Could not generate replacement.
  // CHECK-NEXT: */
  f = ynf(i, f);

  // CHECK: /*
  // CHECK-NEXT: DPCT1004:{{[0-9]+}}: Could not generate replacement.
  // CHECK-NEXT: */
  d = cyl_bessel_i0(d);
  // CHECK: /*
  // CHECK-NEXT: DPCT1004:{{[0-9]+}}: Could not generate replacement.
  // CHECK-NEXT: */
  d = cyl_bessel_i1(d);
  // CHECK: /*
  // CHECK-NEXT: DPCT1004:{{[0-9]+}}: Could not generate replacement.
  // CHECK-NEXT: */
  d = erfcinv(d);
  // CHECK: /*
  // CHECK-NEXT: DPCT1004:{{[0-9]+}}: Could not generate replacement.
  // CHECK-NEXT: */
  d = erfcx(d);
  // CHECK: /*
  // CHECK-NEXT: DPCT1004:{{[0-9]+}}: Could not generate replacement.
  // CHECK-NEXT: */
  d = erfinv(d);
  // CHECK: /*
  // CHECK-NEXT: DPCT1004:{{[0-9]+}}: Could not generate replacement.
  // CHECK-NEXT: */
  d = j0(d);
  // CHECK: /*
  // CHECK-NEXT: DPCT1004:{{[0-9]+}}: Could not generate replacement.
  // CHECK-NEXT: */
  d = j1(d);
  // CHECK: /*
  // CHECK-NEXT: DPCT1004:{{[0-9]+}}: Could not generate replacement.
  // CHECK-NEXT: */
  d = jn(i, d);
  // CHECK: /*
  // CHECK-NEXT: DPCT1004:{{[0-9]+}}: Could not generate replacement.
  // CHECK-NEXT: */
  d = norm(i, &d);
  // CHECK: /*
  // CHECK-NEXT: DPCT1004:{{[0-9]+}}: Could not generate replacement.
  // CHECK-NEXT: */
  d = norm3d(d, d, d);
  // CHECK: /*
  // CHECK-NEXT: DPCT1004:{{[0-9]+}}: Could not generate replacement.
  // CHECK-NEXT: */
  d = norm4d(d, d, d, d);
  // CHECK: /*
  // CHECK-NEXT: DPCT1004:{{[0-9]+}}: Could not generate replacement.
  // CHECK-NEXT: */
  d = normcdf(d);
  // CHECK: /*
  // CHECK-NEXT: DPCT1004:{{[0-9]+}}: Could not generate replacement.
  // CHECK-NEXT: */
  d = normcdfinv(d);
  // CHECK: /*
  // CHECK-NEXT: DPCT1004:{{[0-9]+}}: Could not generate replacement.
  // CHECK-NEXT: */
  d = rcbrt(d);
  // CHECK: /*
  // CHECK-NEXT: DPCT1004:{{[0-9]+}}: Could not generate replacement.
  // CHECK-NEXT: */
  d = rnorm3d(d, d, d);
  // CHECK: /*
  // CHECK-NEXT: DPCT1004:{{[0-9]+}}: Could not generate replacement.
  // CHECK-NEXT: */
  d = rnorm4d(d, d, d, d);
  // CHECK: /*
  // CHECK-NEXT: DPCT1004:{{[0-9]+}}: Could not generate replacement.
  // CHECK-NEXT: */
  d = rnorm(i, &d);
  // CHECK: /*
  // CHECK-NEXT: DPCT1004:{{[0-9]+}}: Could not generate replacement.
  // CHECK-NEXT: */
  d = scalbln(d, l);
  // CHECK: /*
  // CHECK-NEXT: DPCT1004:{{[0-9]+}}: Could not generate replacement.
  // CHECK-NEXT: */
  d = scalbn(d, i);
  // CHECK: /*
  // CHECK-NEXT: DPCT1004:{{[0-9]+}}: Could not generate replacement.
  // CHECK-NEXT: */
  d = y0(d);
  // CHECK: /*
  // CHECK-NEXT: DPCT1004:{{[0-9]+}}: Could not generate replacement.
  // CHECK-NEXT: */
  d = y1(d);
  // CHECK: /*
  // CHECK-NEXT: DPCT1004:{{[0-9]+}}: Could not generate replacement.
  // CHECK-NEXT: */
  d = yn(i, d);

  // CHECK: /*
  // CHECK-NEXT: DPCT1004:{{[0-9]+}}: Could not generate replacement.
  // CHECK-NEXT: */
  f = __saturatef(f);

  // i = __shfl_down_sync(u, h, u, i);
  // i = __shfl_sync(u, h, u, i);
  // i = __shfl_up_sync(u, h, u, i);
  // i = __shfl_xor_sync(u, h, u, i);

  // CHECK: /*
  // CHECK-NEXT: DPCT1004:{{[0-9]+}}: Could not generate replacement.
  // CHECK-NEXT: */
  i = __double2hiint(d);
  // CHECK: /*
  // CHECK-NEXT: DPCT1004:{{[0-9]+}}: Could not generate replacement.
  // CHECK-NEXT: */
  i = __double2loint(d);
  // CHECK: /*
  // CHECK-NEXT: DPCT1004:{{[0-9]+}}: Could not generate replacement.
  // CHECK-NEXT: */
  d = __hiloint2double(i, i);


  // CHECK: /*
  // CHECK-NEXT: DPCT1004:{{[0-9]+}}: Could not generate replacement.
  // CHECK-NEXT: */
  u = __brev(u);
  // CHECK: /*
  // CHECK-NEXT: DPCT1004:{{[0-9]+}}: Could not generate replacement.
  // CHECK-NEXT: */
  ull = __brevll(ull);
  // CHECK: /*
  // CHECK-NEXT: DPCT1004:{{[0-9]+}}: Could not generate replacement.
  // CHECK-NEXT: */
  u = __byte_perm(u, u, u);
  // CHECK: /*
  // CHECK-NEXT: DPCT1004:{{[0-9]+}}: Could not generate replacement.
  // CHECK-NEXT: */
  i = __ffs(i);
  // CHECK: /*
  // CHECK-NEXT: DPCT1004:{{[0-9]+}}: Could not generate replacement.
  // CHECK-NEXT: */
  i = __ffsll(ll);
  // CHECK: /*
  // CHECK-NEXT: DPCT1004:{{[0-9]+}}: Could not generate replacement.
  // CHECK-NEXT: */
  u = __funnelshift_l(u, u, u);
  // CHECK: /*
  // CHECK-NEXT: DPCT1004:{{[0-9]+}}: Could not generate replacement.
  // CHECK-NEXT: */
  u = __funnelshift_lc(u, u, u);
  // CHECK: /*
  // CHECK-NEXT: DPCT1004:{{[0-9]+}}: Could not generate replacement.
  // CHECK-NEXT: */
  u = __funnelshift_r(u, u, u);
  // CHECK: /*
  // CHECK-NEXT: DPCT1004:{{[0-9]+}}: Could not generate replacement.
  // CHECK-NEXT: */
  u = __funnelshift_rc(u, u, u);
  // CHECK: /*
  // CHECK-NEXT: DPCT1004:{{[0-9]+}}: Could not generate replacement.
  // CHECK-NEXT: */
  ll = __mul64hi(ll, ll);
  // CHECK: /*
  // CHECK-NEXT: DPCT1004:{{[0-9]+}}: Could not generate replacement.
  // CHECK-NEXT: */
  i = __rhadd(i, i);
  // CHECK: /*
  // CHECK-NEXT: DPCT1004:{{[0-9]+}}: Could not generate replacement.
  // CHECK-NEXT: */
  u = __sad(i, i, u);
  // CHECK: /*
  // CHECK-NEXT: DPCT1004:{{[0-9]+}}: Could not generate replacement.
  // CHECK-NEXT: */
  u = __uhadd(u, u);
  // CHECK: /*
  // CHECK-NEXT: DPCT1004:{{[0-9]+}}: Could not generate replacement.
  // CHECK-NEXT: */
  u = __umul24(u, u);
  // CHECK: /*
  // CHECK-NEXT: DPCT1004:{{[0-9]+}}: Could not generate replacement.
  // CHECK-NEXT: */
  ull = __umul64hi(ull, ull);
  // CHECK: /*
  // CHECK-NEXT: DPCT1004:{{[0-9]+}}: Could not generate replacement.
  // CHECK-NEXT: */
  u = __umulhi(u, u);
  // CHECK: /*
  // CHECK-NEXT: DPCT1004:{{[0-9]+}}: Could not generate replacement.
  // CHECK-NEXT: */
  u = __urhadd(u, u);
  // CHECK: /*
  // CHECK-NEXT: DPCT1004:{{[0-9]+}}: Could not generate replacement.
  // CHECK-NEXT: */
  u = __usad(u, u, u);
}

__global__ void testSimulation() {
  float f;
  double d;

  // CHECK: /*
  // CHECK-NEXT: DPCT1017:{{[0-9]+}}: The sycl::floor call is used instead of the nearbyintf call. These two calls do not provide exactly the same functionality. Check the potential precision and/or performance issues for the generated code.
  // CHECK-NEXT: */
  // CHECK-NEXT: f = sycl::floor(f + 0.5);
  f = nearbyintf(f);

  // CHECK: /*
  // CHECK-NEXT: DPCT1017:{{[0-9]+}}: The sycl::floor call is used instead of the nearbyint call. These two calls do not provide exactly the same functionality. Check the potential precision and/or performance issues for the generated code.
  // CHECK-NEXT: */
  // CHECK-NEXT: d = sycl::floor(d + 0.5);
  d = nearbyint(d);

  // CHECK: /*
  // CHECK-NEXT: DPCT1017:{{[0-9]+}}: The sycl::hypot call is used instead of the rhypotf call. These two calls do not provide exactly the same functionality. Check the potential precision and/or performance issues for the generated code.
  // CHECK-NEXT: */
  // CHECK-NEXT: f = 1 / sycl::hypot(f, f);
  f = rhypotf(f, f);

  // CHECK: /*
  // CHECK-NEXT: DPCT1017:{{[0-9]+}}: The sycl::sincos call is used instead of the sincospif call. These two calls do not provide exactly the same functionality. Check the potential precision and/or performance issues for the generated code.
  // CHECK-NEXT: */
  // CHECK-NEXT: f = sycl::sincos(f * DPCT_PI_F, sycl::make_ptr<float, sycl::access::address_space::global_space>(&f));
  sincospif(f, &f, &f);

  // CHECK: /*
  // CHECK-NEXT: DPCT1017:{{[0-9]+}}: The sycl::sincos call is used instead of the sincospi call. These two calls do not provide exactly the same functionality. Check the potential precision and/or performance issues for the generated code.
  // CHECK-NEXT: */
  // CHECK-NEXT: d = sycl::sincos(d * DPCT_PI, sycl::make_ptr<double, sycl::access::address_space::global_space>(&d));
  sincospi(d, &d, &d);
}

__global__ void testIntegerFunctions() {
  int i;
  unsigned u;
  long l;
  long long ll;
  unsigned long long ull;

  // CHECK: i = sycl::clz(i);
  // CHECK-NEXT: i = sycl::clz(ll);
  // CHECK-NEXT: i = sycl::hadd(i, i);
  // CHECK-NEXT: i = sycl::mul24(i, i);
  // CHECK-NEXT: i = sycl::mul_hi(i, i);
  // CHECK-NEXT: i = sycl::popcount(u);
  // CHECK-NEXT: i = sycl::popcount(ull);
  i = __clz(i);
  i = __clzll(ll);
  i = __hadd(i, i);
  i = __mul24(i, i);
  i = __mulhi(i, i);
  i = __popc(u);
  i = __popcll(ull);

  // CHECK: sycl::clz((int)u);
  // CHECK-NEXT: sycl::clz((long long)ull);
  // CHECK-NEXT: sycl::hadd((int)u, (int)u);
  // CHECK-NEXT: sycl::mul24((int)u, (int)u);
  // CHECK-NEXT: sycl::mul_hi((int)u, (int)u);
  __clz(u);
  __clzll(ull);
  __hadd(u, u);
  __mul24(u, u);
  __mulhi(u, u);
}

void testTypecasts() {

}

__global__ void testConditionalOperator(float *deviceArrayFloat) {
  float &f0 = *deviceArrayFloat, &f1 = *(deviceArrayFloat + 1),
        &f2 = *(deviceArrayFloat + 2);
  // CHECK: f0 = sycl::fmax(f0 = (f1) > (f1 == 1 ? 0 : -f2) ? sycl::native::divide(sycl::pow(f1, 2.f), f1) : -f1, f1 + f1 < f2
  // CHECK-NEXT:         ? ((f1) > (f1 == 1 ? 0 : -f2) ? sycl::native::divide(sycl::pow(f2, 2.f), f1) : -f1)
  // CHECK-NEXT:         : -f1);
  // CHECK-NEXT: f0 = f1 > f2 ? sycl::native::divide(sycl::pow(f1, 2.f), f1) : f1;
  // CHECK-NEXT: f0 = sycl::fmax(0 ? sycl::native::divide(sycl::pow(f1, 2.f), f1) : f1, f2);
  f0 = fmaxf(
      f0 = (f1) > (f1 == 1 ? 0 : -f2) ? __fdividef(__powf(f1, 2.f), f1) : -f1,
      f1 + f1 < f2
          ? ((f1) > (f1 == 1 ? 0 : -f2) ? __fdividef(__powf(f2, 2.f), f1) : -f1)
          : -f1);
  f0 = f1 > f2 ? __fdividef(__powf(f1, 2.f), f1) : f1;
  f0 = fmaxf(0 ? __fdividef(__powf(f1, 2.f), f1) : f1, f2);
}

int main() {
  testDouble();
  testFloat();
  testTypecasts();
}

// Host max/min functions with integer parameters are in <algorithm> instead of <cmath>, so we need to
// migrate them to std versions and do necessary casts.
// The following migration is to make the resulted code compilable by dpcpp
// rule1: in pure __host__ functions, math functions are migrated to std alternatives
// rule2: math functions in __device__ or __global__ functions are always migrated to sycl alternatives
// rule3: functions in std namespace always remain untouched

// CHECK:  int foo(int i, int j) {
// CHECK-NEXT:   return std::max(i, j) + std::min(i, j);
// CHECK-NEXT: }
__host__ int foo(int i, int j) {
  return max(i, j) + min(i, j);
}

// CHECK:  float foo(float f, float g) {
// CHECK-NEXT:   return fmaxf(f, g) + fminf(f, g);
// CHECK-NEXT: }
__host__ float foo(float f, float g) {
  return max(f, g) + min(f, g);
}

// CHECK:  int foo2(int i, int j) {
// CHECK-NEXT:   return sycl::max(i, j) + sycl::min(i, j);
// CHECK-NEXT: }
__device__ int foo2(int i, int j) {
  return max(i, j) + min(i, j);
}

// CHECK:  float foo2(float f, float g) {
// CHECK-NEXT:   return sycl::max(f, g) + sycl::min(f, g);
// CHECK-NEXT: }
__device__ float foo2(float f, float g) {
  return max(f, g) + min(f, g);
}

// CHECK:  int  foo3(int i, int j) {
// CHECK-NEXT:   return sycl::max(i, j) + sycl::min(i, j);
// CHECK-NEXT: }
__device__ int __host__ foo3(int i, int j) {
  return max(i, j) + min(i, j);
}

// CHECK:  float  foo3(float f, float g) {
// CHECK-NEXT:   return sycl::max(f, g) + sycl::min(f, g);
// CHECK-NEXT: }
__device__ float __host__ foo3(float f, float g) {
  return max(f, g) + min(f, g);
}

// CHECK:  int bar(short i, long j) {
// CHECK-NEXT:   return std::max<long>(i, j) + std::min<long>(i, j);
// CHECK-NEXT: }
__host__ int bar(short i, long j) {
  return max(i, j) + min(i, j);
}

// CHECK:  int bar(unsigned short i, unsigned long j) {
// CHECK-NEXT:   return std::max<unsigned long>(i, j) + std::min<unsigned long>(i, j);
// CHECK-NEXT: }
__host__ int bar(unsigned short i, unsigned long j) {
  return max(i, j) + min(i, j);
}

// CHECK:  int bar(unsigned short i, long j) {
// CHECK-NEXT:   return max(i, j) + min(i, j);
// CHECK-NEXT: }
__host__ int bar(unsigned short i, long j) {
  return max(i, j) + min(i, j);
}

// CHECK:  int bar(long i, unsigned short j) {
// CHECK-NEXT:   return max(i, j) + min(i, j);
// CHECK-NEXT: }
__host__ int bar(long i, unsigned short j) {
  return max(i, j) + min(i, j);
}

// CHECK:  int bar(short i, unsigned long j) {
// CHECK-NEXT:   return std::max<unsigned long>(i, j) + std::min<unsigned long>(i, j);
// CHECK-NEXT: }
__host__ int bar(short i, unsigned long j) {
  return max(i, j) + min(i, j);
}

// CHECK:  int bar(unsigned long i, short j) {
// CHECK-NEXT:   return std::max<unsigned long>(i, j) + std::min<unsigned long>(i, j);
// CHECK-NEXT: }
__host__ int bar(unsigned long i, short j) {
  return max(i, j) + min(i, j);
}

typedef int INT;
typedef unsigned UINT;
using int_t = int;
using uint_t = unsigned;

// CHECK: int foo(UINT i, INT j) {
// CHECK-NEXT:   return std::max<UINT>(i, j) + std::min<UINT>(i, j);
// CHECK-NEXT: }
int foo(UINT i, INT j) {
  return max(i, j) + min(i, j);
}

// CHECK: int foo(INT i, UINT j) {
// CHECK-NEXT:   return std::max<UINT>(i, j) + std::min<UINT>(i, j);
// CHECK-NEXT: }
int foo(INT i, UINT j) {
  return max(i, j) + min(i, j);
}

// CHECK: int bar(uint_t i, int_t j) {
// CHECK-NEXT:   return std::max<uint_t>(i, j) + std::min<uint_t>(i, j);
// CHECK-NEXT: }
int bar(uint_t i, int_t j) {
  return max(i, j) + min(i, j);
}

// CHECK: int bar(int_t i, uint_t j) {
// CHECK-NEXT:   return std::max<uint_t>(i, j) + std::min<uint_t>(i, j);
// CHECK-NEXT: }
int bar(int_t i, uint_t j) {
  return max(i, j) + min(i, j);
}

__device__ void test_pow() {
  int i;
  float f;
  double d;

  // CHECK: sycl::pown((float)i, i);
  pow(i, i);
  // CHECK: sycl::pown(f, i);
  pow(f, i);
  // CHECK: sycl::pown(d, i);
  pow(d, i);

  // CHECK: sycl::pow((float)i, f);
  pow(i, f);
  // CHECK: sycl::pow(f, f);
  pow(f, f);
  // CHECK: sycl::pow(d, (double)f);
  pow(d, f);

  // CHECK: sycl::pow((double)i, d);
  pow(i, d);
  // CHECK: sycl::pow((double)f, d);
  pow(f, d);
  // CHECK: sycl::pow(d, d);
  pow(d, d);
}

__global__ void foobar(int i) {
  // CHECK: sycl::max((unsigned int)i, (unsigned int)item_ct1.get_local_id(2));
  // CHECK-NEXT: sycl::max((unsigned int)i, (unsigned int)item_ct1.get_local_id(1));
  // CHECK-NEXT: sycl::max((unsigned int)i, (unsigned int)item_ct1.get_local_id(0));
  // CHECK-NEXT: sycl::max((unsigned int)item_ct1.get_local_id(2), (unsigned int)i);
  // CHECK-NEXT: sycl::max((unsigned int)item_ct1.get_local_id(1), (unsigned int)i);
  // CHECK-NEXT: sycl::max((unsigned int)item_ct1.get_local_id(0), (unsigned int)i);
  max(i, threadIdx.x);
  max(i, threadIdx.y);
  max(i, threadIdx.z);
  max(threadIdx.x, i);
  max(threadIdx.y, i);
  max(threadIdx.z, i);

  // CHECK: sycl::max((unsigned int)i, (unsigned int)item_ct1.get_group(2));
  // CHECK-NEXT: sycl::max((unsigned int)i, (unsigned int)item_ct1.get_group(1));
  // CHECK-NEXT: sycl::max((unsigned int)i, (unsigned int)item_ct1.get_group(0));
  // CHECK-NEXT: sycl::max((unsigned int)item_ct1.get_group(2), (unsigned int)i);
  // CHECK-NEXT: sycl::max((unsigned int)item_ct1.get_group(1), (unsigned int)i);
  // CHECK-NEXT: sycl::max((unsigned int)item_ct1.get_group(0), (unsigned int)i);
  max(i, blockIdx.x);
  max(i, blockIdx.y);
  max(i, blockIdx.z);
  max(blockIdx.x, i);
  max(blockIdx.y, i);
  max(blockIdx.z, i);

  // CHECK: sycl::max((unsigned int)i, (unsigned int)item_ct1.get_local_range(2));
  // CHECK-NEXT: sycl::max((unsigned int)i, (unsigned int)item_ct1.get_local_range(1));
  // CHECK-NEXT: sycl::max((unsigned int)i, (unsigned int)item_ct1.get_local_range(0));
  // CHECK-NEXT: sycl::max((unsigned int)item_ct1.get_local_range(2), (unsigned int)i);
  // CHECK-NEXT: sycl::max((unsigned int)item_ct1.get_local_range(1), (unsigned int)i);
  // CHECK-NEXT: sycl::max((unsigned int)item_ct1.get_local_range(0), (unsigned int)i);
  max(i, blockDim.x);
  max(i, blockDim.y);
  max(i, blockDim.z);
  max(blockDim.x, i);
  max(blockDim.y, i);
  max(blockDim.z, i);

  // CHECK: sycl::min((unsigned int)i, (unsigned int)item_ct1.get_local_id(2));
  // CHECK-NEXT: sycl::min((unsigned int)i, (unsigned int)item_ct1.get_local_id(1));
  // CHECK-NEXT: sycl::min((unsigned int)i, (unsigned int)item_ct1.get_local_id(0));
  // CHECK-NEXT: sycl::min((unsigned int)item_ct1.get_local_id(2), (unsigned int)i);
  // CHECK-NEXT: sycl::min((unsigned int)item_ct1.get_local_id(1), (unsigned int)i);
  // CHECK-NEXT: sycl::min((unsigned int)item_ct1.get_local_id(0), (unsigned int)i);
  min(i, threadIdx.x);
  min(i, threadIdx.y);
  min(i, threadIdx.z);
  min(threadIdx.x, i);
  min(threadIdx.y, i);
  min(threadIdx.z, i);

  // CHECK: sycl::min((unsigned int)i, (unsigned int)item_ct1.get_group(2));
  // CHECK-NEXT: sycl::min((unsigned int)i, (unsigned int)item_ct1.get_group(1));
  // CHECK-NEXT: sycl::min((unsigned int)i, (unsigned int)item_ct1.get_group(0));
  // CHECK-NEXT: sycl::min((unsigned int)item_ct1.get_group(2), (unsigned int)i);
  // CHECK-NEXT: sycl::min((unsigned int)item_ct1.get_group(1), (unsigned int)i);
  // CHECK-NEXT: sycl::min((unsigned int)item_ct1.get_group(0), (unsigned int)i);
  min(i, blockIdx.x);
  min(i, blockIdx.y);
  min(i, blockIdx.z);
  min(blockIdx.x, i);
  min(blockIdx.y, i);
  min(blockIdx.z, i);

  // CHECK: sycl::min((unsigned int)i, (unsigned int)item_ct1.get_local_range(2));
  // CHECK-NEXT: sycl::min((unsigned int)i, (unsigned int)item_ct1.get_local_range(1));
  // CHECK-NEXT: sycl::min((unsigned int)i, (unsigned int)item_ct1.get_local_range(0));
  // CHECK-NEXT: sycl::min((unsigned int)item_ct1.get_local_range(2), (unsigned int)i);
  // CHECK-NEXT: sycl::min((unsigned int)item_ct1.get_local_range(1), (unsigned int)i);
  // CHECK-NEXT: sycl::min((unsigned int)item_ct1.get_local_range(0), (unsigned int)i);
  min(i, blockDim.x);
  min(i, blockDim.y);
  min(i, blockDim.z);
  min(blockDim.x, i);
  min(blockDim.y, i);
  min(blockDim.z, i);
}

void do_migration() {
  int i, j;
  // CHECK: std::max(i, j);
  max(i, j);
}
__global__ void do_migration2() {
  int i, j;
  // CHECK: sycl::max(i, j);
  max(i, j);
}
__device__ void do_migration3() {
  int i, j;
  // CHECK: sycl::max(i, j);
  max(i, j);
}
__host__ __device__ void do_migration4() {
  int i, j;
  // CHECK: sycl::max(i, j);
  max(i, j);
}
int max(int i, int j) {
  return i > j ? i : j;
}
namespace t {
int max(int i, int j) {
  return i > j ? i : j;
}
}
void no_migration() {
  int i, j;
  // CHECK: max(i, j);
  max(i, j);
}
void no_migration2() {
  int i, j;
  // CHECK: t::max(i, j);
  t::max(i, j);
}
void no_migration3() {
  int i, j;
  // CHECK: std::max(i, j);
  std::max(i, j);
}
__host__ void no_migration4() {
  int i, j;
  // CHECK: max(i, j);
  max(i, j);
}

void ns() {
  using namespace std;
  int i, j;
  // CHECK: max(i, j);
  max(i, j);
}

void no_migration5() {
  float f;
  int i;

  //CHECK: std::max(i, i);
  //CHECK-NEXT: std::min(i, i);
  //CHECK-NEXT: std::fabs(f);
  //CHECK-NEXT: std::frexpf(f, &i);
  //CHECK-NEXT: std::modff(f, &f);
  //CHECK-NEXT: std::nearbyintf(f);
  //CHECK-NEXT: std::remquof(f, f, &i);
  //CHECK-NEXT: std::acosf(f);
  //CHECK-NEXT: std::acoshf(f);
  //CHECK-NEXT: std::asinf(f);
  //CHECK-NEXT: std::asinhf(f);
  //CHECK-NEXT: std::abs(f);
  //CHECK-NEXT: std::frexp(f, &i);
  //CHECK-NEXT: std::modf(f, &f);
  //CHECK-NEXT: std::nearbyint(f);
  //CHECK-NEXT: std::remquo(f, f, &i);
  //CHECK-NEXT: std::acos(f);
  //CHECK-NEXT: std::acosh(f);
  //CHECK-NEXT: std::asin(f);
  //CHECK-NEXT: std::asinh(f);
  std::max(i, i);
  std::min(i, i);
  std::fabs(f);
  std::frexpf(f, &i);
  std::modff(f, &f);
  std::nearbyintf(f);
  std::remquof(f, f, &i);
  std::acosf(f);
  std::acoshf(f);
  std::asinf(f);
  std::asinhf(f);
  std::abs(f);
  std::frexp(f, &i);
  std::modf(f, &f);
  std::nearbyint(f);
  std::remquo(f, f, &i);
  std::acos(f);
  std::acosh(f);
  std::asin(f);
  std::asinh(f);
}

__device__ void do_migration5() {
  float f;
  int i;

  //CHECK: sycl::max(i, i);
  //CHECK-NEXT: sycl::min(i, i);
  //CHECK-NEXT:  sycl::fabs(f);
  //CHECK-NEXT: /*
  //CHECK-NEXT: DPCT1017:{{[0-9]+}}: The sycl::frexp call is used instead of the frexpf call. These two calls do not provide exactly the same functionality. Check the potential precision and/or performance issues for the generated code.
  //CHECK-NEXT: */
  //CHECK-NEXT: sycl::frexp(f, sycl::make_ptr<int, sycl::access::address_space::global_space>(&i));
  //CHECK-NEXT: /*
  //CHECK-NEXT: DPCT1017:{{[0-9]+}}: The sycl::modf call is used instead of the modff call. These two calls do not provide exactly the same functionality. Check the potential precision and/or performance issues for the generated code.
  //CHECK-NEXT: */
  //CHECK-NEXT: sycl::modf(f, sycl::make_ptr<float, sycl::access::address_space::global_space>(&f));
  //CHECK-NEXT: /*
  //CHECK-NEXT: DPCT1017:{{[0-9]+}}: The sycl::floor call is used instead of the nearbyintf call. These two calls do not provide exactly the same functionality. Check the potential precision and/or performance issues for the generated code.
  //CHECK-NEXT: */
  //CHECK-NEXT: sycl::floor(f + 0.5);
  //CHECK-NEXT: /*
  //CHECK-NEXT: DPCT1017:{{[0-9]+}}: The sycl::remquo call is used instead of the remquof call. These two calls do not provide exactly the same functionality. Check the potential precision and/or performance issues for the generated code.
  //CHECK-NEXT: */
  //CHECK-NEXT: sycl::remquo(f, f, sycl::make_ptr<int, sycl::access::address_space::global_space>(&i));
  //CHECK-NEXT: sycl::acos(f);
  //CHECK-NEXT: sycl::acosh(f);
  //CHECK-NEXT: sycl::asin(f);
  //CHECK-NEXT: sycl::asinh(f);
  //CHECK-NEXT: sycl::fabs(f);
  //CHECK-NEXT: /*
  //CHECK-NEXT: DPCT1017:{{[0-9]+}}: The sycl::frexp call is used instead of the frexp call. These two calls do not provide exactly the same functionality. Check the potential precision and/or performance issues for the generated code.
  //CHECK-NEXT: */
  //CHECK-NEXT: sycl::frexp(f, sycl::make_ptr<int, sycl::access::address_space::global_space>(&i));
  //CHECK-NEXT: /*
  //CHECK-NEXT: DPCT1017:{{[0-9]+}}: The sycl::modf call is used instead of the modf call. These two calls do not provide exactly the same functionality. Check the potential precision and/or performance issues for the generated code.
  //CHECK-NEXT: */
  //CHECK-NEXT: sycl::modf(f, sycl::make_ptr<double, sycl::access::address_space::global_space>(&f));
  //CHECK-NEXT: /*
  //CHECK-NEXT: DPCT1017:{{[0-9]+}}: The sycl::floor call is used instead of the nearbyint call. These two calls do not provide exactly the same functionality. Check the potential precision and/or performance issues for the generated code.
  //CHECK-NEXT: */
  //CHECK-NEXT: sycl::floor(f + 0.5);
  //CHECK-NEXT: /*
  //CHECK-NEXT: DPCT1017:{{[0-9]+}}: The sycl::remquo call is used instead of the remquo call. These two calls do not provide exactly the same functionality. Check the potential precision and/or performance issues for the generated code.
  //CHECK-NEXT: */
  //CHECK-NEXT: sycl::remquo(f, f, sycl::make_ptr<int, sycl::access::address_space::global_space>(&i));
  //CHECK-NEXT: sycl::acos(f);
  //CHECK-NEXT: sycl::acosh(f);
  //CHECK-NEXT: sycl::asin(f);
  //CHECK-NEXT: sycl::asinh(f);
  std::max(i, i);
  std::min(i, i);
  std::fabs(f);
  std::frexpf(f, &i);
  std::modff(f, &f);
  std::nearbyintf(f);
  std::remquof(f, f, &i);
  std::acosf(f);
  std::acoshf(f);
  std::asinf(f);
  std::asinhf(f);
  std::abs(f);
  std::frexp(f, &i);
  std::modf(f, &f);
  std::nearbyint(f);
  std::remquo(f, f, &i);
  std::acos(f);
  std::acosh(f);
  std::asin(f);
  std::asinh(f);
}

__global__ void do_migration6() {
  float f;
  int i;

  //CHECK: sycl::max(i, i);
  //CHECK-NEXT: sycl::min(i, i);
  //CHECK-NEXT: sycl::fabs(f);
  //CHECK-NEXT: /*
  //CHECK-NEXT: DPCT1017:{{[0-9]+}}: The sycl::frexp call is used instead of the frexpf call. These two calls do not provide exactly the same functionality. Check the potential precision and/or performance issues for the generated code.
  //CHECK-NEXT: */
  //CHECK-NEXT: sycl::frexp(f, sycl::make_ptr<int, sycl::access::address_space::global_space>(&i));
  //CHECK-NEXT: /*
  //CHECK-NEXT: DPCT1017:{{[0-9]+}}: The sycl::modf call is used instead of the modff call. These two calls do not provide exactly the same functionality. Check the potential precision and/or performance issues for the generated code.
  //CHECK-NEXT: */
  //CHECK-NEXT: sycl::modf(f, sycl::make_ptr<float, sycl::access::address_space::global_space>(&f));
  //CHECK-NEXT: /*
  //CHECK-NEXT: DPCT1017:{{[0-9]+}}: The sycl::floor call is used instead of the nearbyintf call. These two calls do not provide exactly the same functionality. Check the potential precision and/or performance issues for the generated code.
  //CHECK-NEXT: */
  //CHECK-NEXT: sycl::floor(f + 0.5);
  //CHECK-NEXT: /*
  //CHECK-NEXT: DPCT1017:{{[0-9]+}}: The sycl::remquo call is used instead of the remquof call. These two calls do not provide exactly the same functionality. Check the potential precision and/or performance issues for the generated code.
  //CHECK-NEXT: */
  //CHECK-NEXT: sycl::remquo(f, f, sycl::make_ptr<int, sycl::access::address_space::global_space>(&i));
  //CHECK-NEXT: sycl::acos(f);
  //CHECK-NEXT: sycl::acosh(f);
  //CHECK-NEXT: sycl::asin(f);
  //CHECK-NEXT: sycl::asinh(f);
  //CHECK-NEXT: sycl::fabs(f);
  //CHECK-NEXT: /*
  //CHECK-NEXT: DPCT1017:{{[0-9]+}}: The sycl::frexp call is used instead of the frexp call. These two calls do not provide exactly the same functionality. Check the potential precision and/or performance issues for the generated code.
  //CHECK-NEXT: */
  //CHECK-NEXT: sycl::frexp(f, sycl::make_ptr<int, sycl::access::address_space::global_space>(&i));
  //CHECK-NEXT: /*
  //CHECK-NEXT: DPCT1017:{{[0-9]+}}: The sycl::modf call is used instead of the modf call. These two calls do not provide exactly the same functionality. Check the potential precision and/or performance issues for the generated code.
  //CHECK-NEXT: */
  //CHECK-NEXT: sycl::modf(f, sycl::make_ptr<double, sycl::access::address_space::global_space>(&f));
  //CHECK-NEXT: /*
  //CHECK-NEXT: DPCT1017:{{[0-9]+}}: The sycl::floor call is used instead of the nearbyint call. These two calls do not provide exactly the same functionality. Check the potential precision and/or performance issues for the generated code.
  //CHECK-NEXT: */
  //CHECK-NEXT: sycl::floor(f + 0.5);
  //CHECK-NEXT: /*
  //CHECK-NEXT: DPCT1017:{{[0-9]+}}: The sycl::remquo call is used instead of the remquo call. These two calls do not provide exactly the same functionality. Check the potential precision and/or performance issues for the generated code.
  //CHECK-NEXT: */
  //CHECK-NEXT: sycl::remquo(f, f, sycl::make_ptr<int, sycl::access::address_space::global_space>(&i));
  //CHECK-NEXT: sycl::acos(f);
  //CHECK-NEXT: sycl::acosh(f);
  //CHECK-NEXT: sycl::asin(f);
  //CHECK-NEXT: sycl::asinh(f);
  std::max(i, i);
  std::min(i, i);
  std::fabs(f);
  std::frexpf(f, &i);
  std::modff(f, &f);
  std::nearbyintf(f);
  std::remquof(f, f, &i);
  std::acosf(f);
  std::acoshf(f);
  std::asinf(f);
  std::asinhf(f);
  std::abs(f);
  std::frexp(f, &i);
  std::modf(f, &f);
  std::nearbyint(f);
  std::remquo(f, f, &i);
  std::acos(f);
  std::acosh(f);
  std::asin(f);
  std::asinh(f);
}

__device__ __host__ void do_migration7() {
  float f;
  int i;

  //CHECK: sycl::max(i, i);
  //CHECK-NEXT: sycl::min(i, i);
  //CHECK-NEXT: sycl::fabs(f);
  //CHECK-NEXT: /*
  //CHECK-NEXT: DPCT1017:{{[0-9]+}}: The sycl::frexp call is used instead of the frexpf call. These two calls do not provide exactly the same functionality. Check the potential precision and/or performance issues for the generated code.
  //CHECK-NEXT: */
  //CHECK-NEXT: sycl::frexp(f, sycl::make_ptr<int, sycl::access::address_space::global_space>(&i));
  //CHECK-NEXT: /*
  //CHECK-NEXT: DPCT1017:{{[0-9]+}}: The sycl::modf call is used instead of the modff call. These two calls do not provide exactly the same functionality. Check the potential precision and/or performance issues for the generated code.
  //CHECK-NEXT: */
  //CHECK-NEXT: sycl::modf(f, sycl::make_ptr<float, sycl::access::address_space::global_space>(&f));
  //CHECK-NEXT: /*
  //CHECK-NEXT: DPCT1017:{{[0-9]+}}: The sycl::floor call is used instead of the nearbyintf call. These two calls do not provide exactly the same functionality. Check the potential precision and/or performance issues for the generated code.
  //CHECK-NEXT: */
  //CHECK-NEXT: sycl::floor(f + 0.5);
  //CHECK-NEXT: /*
  //CHECK-NEXT: DPCT1017:{{[0-9]+}}: The sycl::remquo call is used instead of the remquof call. These two calls do not provide exactly the same functionality. Check the potential precision and/or performance issues for the generated code.
  //CHECK-NEXT: */
  //CHECK-NEXT: sycl::remquo(f, f, sycl::make_ptr<int, sycl::access::address_space::global_space>(&i));
  //CHECK-NEXT: sycl::acos(f);
  //CHECK-NEXT: sycl::acosh(f);
  //CHECK-NEXT: sycl::asin(f);
  //CHECK-NEXT: sycl::asinh(f);
  //CHECK-NEXT: sycl::fabs(f);
  //CHECK-NEXT: /*
  //CHECK-NEXT: DPCT1017:{{[0-9]+}}: The sycl::frexp call is used instead of the frexp call. These two calls do not provide exactly the same functionality. Check the potential precision and/or performance issues for the generated code.
  //CHECK-NEXT: */
  //CHECK-NEXT: sycl::frexp(f, sycl::make_ptr<int, sycl::access::address_space::global_space>(&i));
  //CHECK-NEXT: /*
  //CHECK-NEXT: DPCT1017:{{[0-9]+}}: The sycl::modf call is used instead of the modf call. These two calls do not provide exactly the same functionality. Check the potential precision and/or performance issues for the generated code.
  //CHECK-NEXT: */
  //CHECK-NEXT: sycl::modf(f, sycl::make_ptr<double, sycl::access::address_space::global_space>(&f));
  //CHECK-NEXT: /*
  //CHECK-NEXT: DPCT1017:{{[0-9]+}}: The sycl::floor call is used instead of the nearbyint call. These two calls do not provide exactly the same functionality. Check the potential precision and/or performance issues for the generated code.
  //CHECK-NEXT: */
  //CHECK-NEXT: sycl::floor(f + 0.5);
  //CHECK-NEXT: /*
  //CHECK-NEXT: DPCT1017:{{[0-9]+}}: The sycl::remquo call is used instead of the remquo call. These two calls do not provide exactly the same functionality. Check the potential precision and/or performance issues for the generated code.
  //CHECK-NEXT: */
  //CHECK-NEXT: sycl::remquo(f, f, sycl::make_ptr<int, sycl::access::address_space::global_space>(&i));
  //CHECK-NEXT: sycl::acos(f);
  //CHECK-NEXT: sycl::acosh((double)f);
  //CHECK-NEXT: sycl::asin(f);
  //CHECK-NEXT: sycl::asinh((double)f);
  std::max(i, i);
  std::min(i, i);
  std::fabs(f);
  std::frexpf(f, &i);
  std::modff(f, &f);
  std::nearbyintf(f);
  std::remquof(f, f, &i);
  std::acosf(f);
  std::acoshf(f);
  std::asinf(f);
  std::asinhf(f);
  std::abs(f);
  std::frexp(f, &i);
  std::modf(f, &f);
  std::nearbyint(f);
  std::remquo(f, f, &i);
  std::acos(f);
  std::acosh(f);
  std::asin(f);
  std::asinh(f);
}

// crt wrapper API
__device__ void foo1() {
  int i;
  float f;
  unsigned int ui;
  long long ll;
  unsigned long long ull;

  // CHECK: i = sycl::mul24(i, i);
  i = mul24(i, i);
}

__device__ void test_recursive_unary() {
  int i, j, k;
  // CHECK: sycl::max(-sycl::max(-sycl::abs(i), j), k);
  max(-max(-abs(i), j), k);
}

__device__ void do_math(int i, int j) {
  // CHECK: sycl::sqrt((float)i);
  sqrtf(i);
  // CHECK: sycl::sqrt((double)i);
  sqrt(i);
  // CHECK: sycl::fmod((double)i, (double)j);
  fmod(i, j);
  // CHECK: sycl::sin((double)i);
  sin(i);
  // CHECK: sycl::cos((double)i);
  cos(i);
}

__device__ void do_math(float i, float j) {
  // CHECK: sycl::sqrt(i);
  sqrtf(i);
  // CHECK: sycl::sqrt(i);
  sqrt(i);
  // CHECK: sycl::fmod(i, j);
  fmod(i, j);
  // CHECK: sycl::sin(i);
  sin(i);
  // CHECK: sycl::cos(i);
  cos(i);
}

__device__ void do_math(double i, double j) {
  // CHECK: sycl::sqrt((float)i);
  sqrtf(i);
  // CHECK: sycl::sqrt(i);
  sqrt(i);
  // CHECK: sycl::fmod(i, j);
  fmod(i, j);
  // CHECK: sycl::sin(i);
  sin(i);
  // CHECK: sycl::cos(i);
  cos(i);
}
