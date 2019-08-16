/******************************************************************************
* INTEL CONFIDENTIAL
*
* Copyright 2018 - 2019 Intel Corporation.
*
* This software and the related documents are Intel copyrighted materials,
* and your use of them is governed by the express license under which they
* were provided to you ("License"). Unless the License provides otherwise,
* you may not use, modify, copy, publish, distribute, disclose or transmit
* this software or the related documents without Intel's prior written
* permission.

* This software and the related documents are provided as is, with no express
* or implied warranties, other than those that are expressly stated in the
* License.
*****************************************************************************/

//===--- dpct_kernel.hpp ------------------------------*- C++ -*---===//


#ifndef DPCT_KERNEL_H
#define DPCT_KERNEL_H

#include <CL/sycl.hpp>

struct dpct_kernel_function_info {
  int max_work_group_size = 0;
};

static void get_kernel_function_info(dpct_kernel_function_info *kernel_info,
                                     const void *function) {
  kernel_info->max_work_group_size =
      dpct::device_manager()
          .current_device()
          .get_info<cl::sycl::info::device::max_work_group_size>();
}

#endif // !DPCT_KERNEL_H
