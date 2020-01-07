/******************************************************************************
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

//===--- device.hpp ------------------------------*- C++ -*---===//

#ifndef __DPCT_DEVICE_HPP__
#define __DPCT_DEVICE_HPP__

#include <CL/sycl.hpp>
#include <cstring>
#include <iostream>
#include <set>
#include <sstream>

namespace dpct {

enum class compute_mode { default_, exclusive, prohibited, exclusive_process };

/// DPC++ default exception handler
auto exception_handler = [](cl::sycl::exception_list exceptions) {
  for (std::exception_ptr const &e : exceptions) {
    try {
      std::rethrow_exception(e);
    } catch (cl::sycl::exception const &e) {
      std::cerr << "Caught asynchronous SYCL exception:" << std::endl
                << e.what() << std::endl
                << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__
                << std::endl;
    }
  }
};

/// Device info
class device_info {
public:
  // get interface
  char *get_name() { return _name; }
  cl::sycl::id<3> get_max_work_item_sizes() { return _max_work_item_sizes; }
  bool get_host_unified_memory() { return _host_unified_memory; }
  int get_major_version() { return _major; }
  int get_minor_version() { return _minor; }
  int get_integrated() { return _integrated; }
  int get_max_clock_frequency() { return _frequency; }
  int get_max_compute_units() { return _max_compute_units; }
  int get_max_work_group_size() { return _max_work_group_size; }
  int get_max_sub_group_size() { return _max_sub_group_size; }
  int get_max_work_items_per_compute_unit() {
    return _max_work_items_per_compute_unit;
  }
  size_t *get_max_nd_range_size() { return _max_nd_range_size; }
  size_t get_global_mem_size() { return _global_mem_size; }
  size_t get_local_mem_size() { return _local_mem_size; }
  compute_mode get_mode() { return _compute_mode; }
  // set interface
  void set_name(const char *name) { std::strncpy(_name, name, 256); }
  void set_max_work_item_sizes(const cl::sycl::id<3> max_work_item_sizes) {
    _max_work_item_sizes = max_work_item_sizes;
  }
  void set_host_unified_memory(bool host_unified_memory) {
    _host_unified_memory = host_unified_memory;
  }
  void set_major_version(int major) { _major = major; }
  void set_minor_version(int minor) { _minor = minor; }
  void set_integrated(int integrated) { _integrated = integrated; }
  void set_max_clock_frequency(int frequency) { _frequency = frequency; }
  void set_max_compute_units(int max_compute_units) {
    _max_compute_units = max_compute_units;
  }
  void set_global_mem_size(size_t global_mem_size) {
    _global_mem_size = global_mem_size;
  }
  void set_local_mem_size(size_t local_mem_size) {
    _local_mem_size = local_mem_size;
  }
  void set_mode(compute_mode compute_mode) { _compute_mode = compute_mode; }
  void set_max_work_group_size(int max_work_group_size) {
    _max_work_group_size = max_work_group_size;
  }
  void set_max_sub_group_size(int max_sub_group_size) {
    _max_sub_group_size = max_sub_group_size;
  }
  void
  set_max_work_items_per_compute_unit(int max_work_items_per_compute_unit) {
    _max_work_items_per_compute_unit = max_work_items_per_compute_unit;
  }
  void set_max_nd_range_size(int max_nd_range_size[]) {
    for (int i = 0; i < 3; i++)
      _max_nd_range_size[i] = max_nd_range_size[i];
  }

private:
  char _name[256];
  cl::sycl::id<3> _max_work_item_sizes;
  bool _host_unified_memory = false;
  int _major;
  int _minor;
  int _integrated = 0;
  int _frequency;
  int _max_compute_units;
  int _max_work_group_size;
  int _max_sub_group_size;
  int _max_work_items_per_compute_unit;
  size_t _global_mem_size;
  size_t _local_mem_size;
  size_t _max_nd_range_size[3];
  compute_mode _compute_mode = compute_mode::default_;
};

/// dpct device extension
class device_ext : public cl::sycl::device {
public:
  device_ext() : cl::sycl::device() {}
  device_ext(const cl::sycl::device &base) : cl::sycl::device(base) {
    _default_queue = cl::sycl::queue(base, exception_handler);
  }

  int is_native_atomic_supported() { return 0; }
  int get_major_version() {
    int major, minor;
    get_version(major, minor);
    return major;
  }

  void get_device_info(device_info &out) {
    device_info prop;
    prop.set_name(get_info<cl::sycl::info::device::name>().c_str());

    int major, minor;
    get_version(major, minor);
    prop.set_major_version(major);
    prop.set_minor_version(minor);

    prop.set_max_work_item_sizes(
        get_info<cl::sycl::info::device::max_work_item_sizes>());
    prop.set_host_unified_memory(
        get_info<cl::sycl::info::device::host_unified_memory>());
    prop.set_max_clock_frequency(
        get_info<cl::sycl::info::device::max_clock_frequency>());
    prop.set_max_compute_units(
        get_info<cl::sycl::info::device::max_compute_units>());
    prop.set_max_work_group_size(
        get_info<cl::sycl::info::device::max_work_group_size>());
    prop.set_global_mem_size(
        get_info<cl::sycl::info::device::global_mem_size>());
    prop.set_local_mem_size(get_info<cl::sycl::info::device::local_mem_size>());

    // For Intel(R) oneAPI DPC++ Compiler, if current device does not support
    // "cl_intel_required_subgroup_size" extension, max_sub_group_size will be
    // initialized to one.
    // For other compilers, just initialize max_sub_group_size to one.
    // This code may need to be updated depending on subgroup support by other
    // compilers.
    size_t max_sub_group_size = 1;
#ifdef __SYCL_COMPILER_VERSION
    if (has_extension("cl_intel_required_subgroup_size")) {
      cl::sycl::vector_class<size_t> sub_group_sizes =
          get_info<cl::sycl::info::device::sub_group_sizes>();
      cl::sycl::vector_class<size_t>::const_iterator max_iter =
          std::max_element(sub_group_sizes.begin(), sub_group_sizes.end());
      max_sub_group_size = *max_iter;
    }
#endif
    prop.set_max_sub_group_size(max_sub_group_size);

    prop.set_max_work_items_per_compute_unit(
        get_info<cl::sycl::info::device::max_work_group_size>());
    int max_nd_range_size[] = {0x7FFFFFFF, 0x7FFFFFFF, 0x7FFFFFFF};
    prop.set_max_nd_range_size(max_nd_range_size);

    out = prop;
  }

  void reset() {
    for (auto q : _queues) {
      // The destructor waits for all commands executing on the queue to
      // complete. It isn't possible to destroy a queue immediately. This is a
      // synchronization point in SYCL.
      q.~queue();
    }
    _queues.clear();
  }

  cl::sycl::queue &default_queue() { return _default_queue; }

  void queues_wait_and_throw() {
    _default_queue.wait_and_throw();
    for (auto q : _queues) {
      q.wait_and_throw();
    }
  }

private:
  void get_version(int &major, int &minor) {
    // Version string has the following format:
    // OpenCL<space><major.minor><space><vendor-specific-information>
    std::stringstream ver;
    ver << get_info<cl::sycl::info::device::version>();
    std::string item;
    std::getline(ver, item, ' '); // OpenCL
    std::getline(ver, item, '.'); // major
    major = std::stoi(item);
    std::getline(ver, item, ' '); // minor
    minor = std::stoi(item);
  }
  cl::sycl::queue _default_queue;
  std::set<cl::sycl::queue> _queues;
};

/// device manager
class device_manager {
public:
  device_manager() {
    cl::sycl::device default_device =
        cl::sycl::device(cl::sycl::default_selector{});
    _devs.push_back(device_ext(default_device));

    std::vector<cl::sycl::device> sycl_all_devs =
        cl::sycl::device::get_devices(cl::sycl::info::device_type::all);
    for (auto &dev : sycl_all_devs) {
      if (dev.get_info<cl::sycl::info::device::vendor_id>() !=
          default_device.get_info<cl::sycl::info::device::vendor_id>()) {
        _devs.push_back(device_ext(dev));
      }
    }
  }
  device_ext &current_device() {
    check_id(_current_device);
    return _devs[_current_device];
  }
  device_ext get_device(unsigned int id) const {
    check_id(id);
    return _devs[id];
  }
  unsigned int current_device_id() const { return _current_device; }
  void select_device(unsigned int id) {
    check_id(id);
    _current_device = id;
  }
  unsigned int device_count() { return _devs.size(); }

private:
  void check_id(unsigned int id) const {
    if (id >= _devs.size()) {
      throw std::string("invalid device id");
    }
  }
  std::vector<device_ext> _devs;
  unsigned int _current_device = 0;
};

/// Util function to get the instance of dpct device manager.
static device_manager &get_device_manager() {
  static device_manager d_m;
  return d_m;
}

/// Util function to get the defualt queue of current device in
/// dpct device manager.
static inline cl::sycl::queue &get_default_queue() {
  return dpct::get_device_manager().current_device().default_queue();
}

/// Util function to get the defualt queue of current device in
/// dpct device manager. Wait till all the tasks in the queue are done.
static inline cl::sycl::queue &get_default_queue_wait() {
  auto &q = get_default_queue();
  q.wait();
  return q;
}

/// Util function to get the current device.
static inline cl::sycl::device &get_current_device() {
  return dpct::get_device_manager().current_device();
}

/// Util function to get the context of the default queue of current
/// device in dpct device manager.
static inline cl::sycl::context get_default_context() {
  return dpct::get_default_queue().get_context();
}

} // namespace dpct

#endif // __DPCT_DEVICE_HPP__
