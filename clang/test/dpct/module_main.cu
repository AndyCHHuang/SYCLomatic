// UNSUPPORTED: -windows-
// RUN: dpct --format-range=none -out-root %T/module_main %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: FileCheck %s --match-full-lines --input-file %T/module_main/module_main.dp.cpp

//CHECK: #include <dlfcn.h>
#include <string>
int main(){
    //CHECK: void * M;
    CUmodule M;
    //CHECK: dpct::kernel_functor F;
    CUfunction F;
    std::string Path, FunctionName, Data;
    //CHECK: /*
    //CHECK-NEXT: DPCT1079:{{[0-9]+}}: You need to replace the "PlaceHolder" with the file name of the dynamic library.
    //CHECK-NEXT: */
    //CHECK-NEXT: M = dlopen(PlaceHolder/*Fix the module file name manually*/, RTLD_LAZY);
    cuModuleLoad(&M, Path.c_str());
    //CHECK: /*
    //CHECK-NEXT: DPCT1079:{{[0-9]+}}: You need to replace the "PlaceHolder" with the file name of the dynamic library.
    //CHECK-NEXT: */
    //CHECK-NEXT: M = dlopen(PlaceHolder/*Fix the module file name manually*/, RTLD_LAZY);
    cuModuleLoadData(&M, Data.c_str());
    //CHECK: F = (dpct::kernel_functor)dlsym(M, (std::string(FunctionName.c_str()) + "_wrapper").c_str());
    cuModuleGetFunction(&F, M, FunctionName.c_str());

    int sharedSize;
    CUstream s;
    void **param, **extra;
    //CHECK:  F(*s, sycl::nd_range<3>(sycl::range<3>(32, 16, 1) * sycl::range<3>(64, 32, 4), sycl::range<3>(64, 32, 4)), sharedSize, param, extra);
    cuLaunchKernel(F, 1, 16, 32, 4, 32, 64, sharedSize, s, param, extra);
    //CHECK:  F(q_ct1, sycl::nd_range<3>(sycl::range<3>(32, 16, 1) * sycl::range<3>(64, 32, 4), sycl::range<3>(64, 32, 4)), sharedSize, param, extra);
    cuLaunchKernel(F, 1, 16, 32, 4, 32, 64, sharedSize, 0, param, extra);
    //CHECK:  F(q_ct1, sycl::nd_range<3>(sycl::range<3>(32, 16, 1) * sycl::range<3>(64, 32, 4), sycl::range<3>(64, 32, 4)), sharedSize, param, extra);
    cuLaunchKernel(F, 1, 16, 32, 4, 32, 64, sharedSize, CU_STREAM_LEGACY, param, extra);

    //CHECK: dpct::image_wrapper_base_p tex;
    //CHECK: tex = (dpct::image_wrapper_base_p)dlsym(M, "tex");
    CUtexref tex;
    cuModuleGetTexRef(&tex, M, "tex");

    //CHECK: dlclose(M);
    cuModuleUnload(M);
    return 0;
}