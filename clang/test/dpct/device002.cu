// RUN: dpct -out-root %T %s -- -x cuda --cuda-host-only --cuda-path="%cuda-path"
// RUN: FileCheck %s --match-full-lines --input-file %T/device002.dp.cpp

#include <stdio.h>

void checkError(cudaError_t err) {

}

int main(int argc, char **argv)
{
int devID = atoi(argv[1]);
cudaDeviceProp cdp;
// CHECK:/*
// CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
// CHECK-NEXT:*/
// CHECK-NEXT: int error_code = (dpct::get_device_manager().get_device(devID).get_device_info(cdp), 0);
cudaError_t error_code = cudaGetDeviceProperties(&cdp, devID);

if (error_code == cudaSuccess) {
// CHECK: /*
// CHECK-NEXT:  DPCT1005:{{[0-9]+}}: The device version is different. You need to rewrite this code.
// CHECK-NEXT: */
// CHECK-NEXT: /*
// CHECK-NEXT:  DPCT1006:{{[0-9]+}}: DPC++ does not provide a standard API to differentiate between integrated/ discrete GPU devices.
// CHECK-NEXT: */
// CHECK-NEXT:if (cdp.get_major_version() < 3 && cdp.get_integrated() != 1) {
    if (cdp.major < 3 && cdp.integrated != 1) {
            printf("do_complex_compute requires compute capability 3.0 or later and not integrated\n");
    }
}

int deviceCount = 0;
// CHECK: deviceCount = dpct::get_device_manager().device_count();
cudaGetDeviceCount(&deviceCount);

int dev_id;
// CHECK: dev_id = dpct::get_device_manager().current_device_id();
cudaGetDevice(&dev_id);

cudaDeviceProp deviceProp;
// CHECK: dpct::get_device_manager().get_device(0).get_device_info(deviceProp);
cudaGetDeviceProperties(&deviceProp, 0);

int atomicSupported;
// CHECK: atomicSupported = dpct::get_device_manager().get_device(dev_id).is_native_atomic_supported();
cudaDeviceGetAttribute(&atomicSupported, cudaDevAttrHostNativeAtomicSupported, dev_id);

int val;
// CHECK: val = dpct::get_device_manager().get_device(dev_id).get_major_version();
cudaDeviceGetAttribute(&val, cudaDevAttrComputeCapabilityMajor, dev_id);

int device1 = 0;
int device2 = 1;
int perfRank = 0;
int accessSupported = 0;

// CHECK:/*
// CHECK-NEXT:DPCT1004:{{[0-9]+}}: Could not generate replacement.
// CHECK-NEXT:*/
// CHECK-NEXT: accessSupported = 0;
cudaDeviceGetP2PAttribute(&accessSupported, cudaDevP2PAttrAccessSupported, device1, device2);

// CHECK:/*
// CHECK-NEXT:DPCT1004:{{[0-9]+}}: Could not generate replacement.
// CHECK-NEXT:*/
// CHECK-NEXT: perfRank = 0;
cudaDeviceGetP2PAttribute(&perfRank, cudaDevP2PAttrPerformanceRank, device1, device2);

// CHECK:/*
// CHECK-NEXT:DPCT1004:{{[0-9]+}}: Could not generate replacement.
// CHECK-NEXT:*/
// CHECK-NEXT: atomicSupported = 0;
cudaDeviceGetP2PAttribute(&atomicSupported, cudaDevP2PAttrNativeAtomicSupported, device1, device2);


char pciBusId[80];
// CHECK:/*
// CHECK-NEXT:DPCT1004:{{[0-9]+}}: Could not generate replacement.
// CHECK-NEXT:*/
cudaDeviceGetPCIBusId(pciBusId, 80, 0);


// CHECK: dpct::get_device_manager().current_device().reset();
cudaDeviceReset();

// CHECK:/*
// CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
// CHECK-NEXT:*/
// CHECK-NEXT:error_code = (dpct::get_device_manager().current_device().reset(), 0);
error_code = cudaDeviceReset();

// CHECK: dpct::get_device_manager().current_device().reset();
cudaThreadExit();

// CHECK:/*
// CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
// CHECK-NEXT:*/
// CHECK-NEXT:error_code = (dpct::get_device_manager().current_device().reset(), 0);
error_code = cudaThreadExit();

// CHECK:/*
// CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
// CHECK-NEXT:*/
// CHECK-NEXT:error_code = (dpct::get_device_manager().select_device(device2), 0);
error_code = cudaSetDevice(device2);
// CHECK: dpct::get_device_manager().select_device(device2);
cudaSetDevice(device2);

// CHECK:dpct::get_device_manager().current_device().queues_wait_and_throw();
// CHECK-NEXT:/*
// CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
// CHECK-NEXT:*/
// CHECK-NEXT:int err = (dpct::get_device_manager().current_device().queues_wait_and_throw(), 0);
// CHECK-NEXT:/*
// CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
// CHECK-NEXT:*/
// CHECK-NEXT:checkError((dpct::get_device_manager().current_device().queues_wait_and_throw(), 0));
// CHECK-NEXT:/*
// CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
// CHECK-NEXT:*/
// CHECK-NEXT:return (dpct::get_device_manager().current_device().queues_wait_and_throw(), 0);
cudaDeviceSynchronize();
cudaError_t err = cudaDeviceSynchronize();
checkError(cudaDeviceSynchronize());
return cudaDeviceSynchronize();
// CHECK:/*
// CHECK-NEXT:DPCT1010:{{[0-9]+}}: SYCL uses exceptions to report errors and does not use the error codes. The call was replaced with 0. You need to rewrite this code.
// CHECK-NEXT:*/
// CHECK-NEXT: int e = 0;
int e = cudaGetLastError();
// CHECK:/*
// CHECK-NEXT:DPCT1010:{{[0-9]+}}: SYCL uses exceptions to report errors and does not use the error codes. The call was replaced with 0. You need to rewrite this code.
// CHECK-NEXT:*/
// CHECK-NEXT: int e1 = 0;
// CHECK-NEXT:/*
// CHECK-NEXT:DPCT1010:{{[0-9]+}}: SYCL uses exceptions to report errors and does not use the error codes. The call was replaced with 0. You need to rewrite this code.
// CHECK-NEXT:*/
// CHECK-NEXT: 0;
int e1 = cudaPeekAtLastError();
cudaPeekAtLastError();
// CHECK:dpct::get_device_manager().current_device().queues_wait_and_throw();
cudaThreadSynchronize();
return 0;
}
