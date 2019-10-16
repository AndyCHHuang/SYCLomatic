// RUN: dpct --usm-level=none -out-root %T %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only -std=c++14
// RUN: FileCheck --input-file %T/texture.dp.cpp --match-full-lines %s

// CHECK: dpct::dpct_image<cl::sycl::float4, 2> tex42;
static texture<float4, 2> tex42;
// CHECK: dpct::dpct_image<cl::sycl::uint2, 1> tex21;
static texture<uint2, 1> tex21;
/// TODO: Expect to support 3D array in future.
// TODO-CHECK: dpct::dpct_image<int, 3> tex13;
// static texture<int, 3> tex13;

// CHECK: void device01(dpct::dpct_image_accessor<cl::sycl::uint2, 1> tex21) {
// CHECK-NEXT: cl::sycl::uint2 u21 = dpct::dpct_read_image(tex21, 1.0f);
// CHECK-NEXT: cl::sycl::uint2 u21_fetch = dpct::dpct_read_image(tex21, 1);
__device__ void device01() {
  uint2 u21 = tex1D(tex21, 1.0f);
  uint2 u21_fetch = tex1Dfetch(tex21, 1);
}

// CHECK: void kernel(dpct::dpct_image_accessor<cl::sycl::float4, 2> tex42, dpct::dpct_image_accessor<cl::sycl::uint2, 1> tex21) {
// CHECK-NEXT: device01(tex21);
// CHECK-NEXT: cl::sycl::float4 f42 = dpct::dpct_read_image(tex42, 1.0f, 1.0f);
/// Texture accessors should be passed down to __global__/__device__ function if used.
__global__ void kernel() {
  device01();
  float4 f42 = tex2D(tex42, 1.0f, 1.0f);
}

int main() {

  // CHECK: cl::sycl::float4 *d_data42;
  // CHECK-NEXT: dpct::dpct_matrix_p a42;
  // CHECK-NEXT: dpct::dpct_malloc(&d_data42, sizeof(cl::sycl::float4) * 32 * 32);
  // CHECK-NEXT: dpct::dpct_image_channel desc42 = dpct::create_image_channel(32, 32, 32, 32, dpct::channel_float);
  // CHECK-NEXT: dpct::dpct_malloc_matrix(&a42, &desc42, 32, 32);
  // CHECK-NEXT: dpct::dpct_memcpy_to_matrix(a42, 0, 0, d_data42, 32 * 32 * sizeof(cl::sycl::float4));
  // CHECK-NEXT: tex42.addr_mode() = cl::sycl::addressing_mode::clamp_to_edge;
  // CHECK-NEXT: tex42.addr_mode() = cl::sycl::addressing_mode::clamp_to_edge;
  // CHECK-NEXT: tex42.addr_mode() = cl::sycl::addressing_mode::clamp_to_edge;
  // CHECK-NEXT: tex42.filter_mode() = cl::sycl::filtering_mode::nearest;
  // CHECK-NEXT: dpct::dpct_attach_image(tex42, a42);
  float4 *d_data42;
  cudaArray_t a42;
  cudaMalloc(&d_data42, sizeof(float4) * 32 * 32);
  cudaChannelFormatDesc desc42 = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);
  cudaMallocArray(&a42, &desc42, 32, 32);
  cudaMemcpyToArray(a42, 0, 0, d_data42, 32 * 32 * sizeof(float4), cudaMemcpyDeviceToDevice);
  tex42.addressMode[0] = cudaAddressModeClamp;
  tex42.addressMode[1] = cudaAddressModeClamp;
  tex42.addressMode[2] = cudaAddressModeClamp;
  tex42.filterMode = cudaFilterModePoint;
  cudaBindTextureToArray(tex42, a42, desc42);

  // CHECK: cl::sycl::uint2 *d_data21;
  // CHECK-NEXT: dpct::dpct_malloc(&d_data21, sizeof(cl::sycl::uint2) * 32);
  // CHECK-NEXT: dpct::dpct_image_channel desc21 = dpct::create_image_channel(32, 32, 0, 0, dpct::channel_unsigned);
  // CHECK-NEXT: tex21.addr_mode() = cl::sycl::addressing_mode::clamp_to_edge;
  // CHECK-NEXT: tex21.addr_mode() = cl::sycl::addressing_mode::clamp_to_edge;
  // CHECK-NEXT: tex21.addr_mode() = cl::sycl::addressing_mode::clamp_to_edge;
  // CHECK-NEXT: tex21.filter_mode() = cl::sycl::filtering_mode::linear;
  // CHECK-NEXT: dpct::dpct_attach_image(tex21, d_data21, desc21, 32 * sizeof(cl::sycl::uint2));
  uint2 *d_data21;
  cudaMalloc(&d_data21, sizeof(uint2) * 32);
  cudaChannelFormatDesc desc21 = cudaCreateChannelDesc(32, 32, 0, 0, cudaChannelFormatKindUnsigned);
  tex21.addressMode[0] = cudaAddressModeClamp;
  tex21.addressMode[1] = cudaAddressModeClamp;
  tex21.addressMode[2] = cudaAddressModeClamp;
  tex21.filterMode = cudaFilterModeLinear;
  cudaBindTexture(0, tex21, d_data21, desc21, 32 * sizeof(uint2));

  // CHECK:   {
  // CHECK-NEXT:   dpct::get_default_queue().submit(
  // CHECK-NEXT:       [&](cl::sycl::handler &cgh) {
  // CHECK-NEXT:         auto tex42_acc = tex42.get_access(cgh);
  // CHECK-NEXT:         auto tex21_acc = tex21.get_access(cgh);
  // CHECK-NEXT:         cgh.parallel_for<dpct_kernel_name<class kernel_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:             cl::sycl::nd_range<3>((cl::sycl::range<3>(1, 1, 1) * cl::sycl::range<3>(1, 1, 1)), cl::sycl::range<3>(1, 1, 1)),
  // CHECK-NEXT:             [=](cl::sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:               kernel(tex42_acc, tex21_acc);
  // CHECK-NEXT:             });
  // CHECK-NEXT:       });
  // CHECK-NEXT: }
  kernel<<<1, 1>>>();

  // CHECK: dpct::dpct_detach_image(tex42);
  // CHECK-NEXT: dpct::dpct_detach_image(tex21);
  cudaUnbindTexture(tex42);
  cudaUnbindTexture(tex21);

  // CHECK: dpct::dpct_free(a42);
  cudaFreeArray(a42);
 
  // CHECK: dpct::dpct_free(d_data42);
  // CHECK-NEXT: dpct::dpct_free(d_data21);
  cudaFree(d_data42);
  cudaFree(d_data21);
}
