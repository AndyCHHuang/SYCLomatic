/******************************************************************************
* INTEL CONFIDENTIAL
*
* Copyright 2018-2019 Intel Corporation.
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

//===--- syclct_device.hpp ------------------------------*- C++ -*---===//

#ifndef SYCLCT_DEVICE_H
#define SYCLCT_DEVICE_H

#include <CL/sycl.hpp>
#include <cstring>
#include <iostream>
#include <set>
#include <sstream>

namespace syclct {

enum class compute_mode { default_, exclusive, prohibited, exclusive_process };

auto exception_handler = [](cl::sycl::exception_list exceptions) {
  for (std::exception_ptr const &e : exceptions) {
    try {
      std::rethrow_exception(e);
    } catch (cl::sycl::exception const &e) {
      std::cerr << "Caught asynchronous SYCL exception:\n"
                << e.what() << std::endl;
    }
  }
};

class sycl_device_info {
public:
  //get interface
  char *get_name() { return _name; }
  cl::sycl::id<3> get_max_work_item_sizes() { return _max_work_item_sizes; }
  bool get_host_unified_memory() { return _host_unified_memory; }
  int get_major_version() { return _major; }
  int get_minor_version() { return _minor; }
  int get_integrated() { return _integrated; }
  int get_max_clock_frequency() { return _frequency; }
  int get_max_compute_units() { return _max_compute_units; }
  int get_max_work_group_size() { return _max_work_group_size; }
  int get_warp_size() { return _warp_size; }
  int get_max_work_items_per_compute_unit() { return _max_work_items_per_compute_unit; }
  size_t *get_max_grid_size() { return _max_grid_size; }
  size_t get_global_mem_size() { return _global_mem_size; }
  size_t get_local_mem_size() { return _local_mem_size; }
  compute_mode get_mode() { return _compute_mode; }
  // set interface
  void set_name(const char* name) {std::strncpy(_name, name,256);}
  void set_max_work_item_sizes(const cl::sycl::id<3> max_work_item_sizes) {_max_work_item_sizes=max_work_item_sizes;}
  void set_host_unified_memory(bool host_unified_memory) {_host_unified_memory=host_unified_memory;}
  void set_major_version(int major) {_major=major;}
  void set_minor_version(int minor) {_minor=minor;}
  void set_integrated(int integrated) {_integrated=integrated;}
  void set_max_clock_frequency(int frequency) {_frequency=frequency;}
  void set_max_compute_units(int max_compute_units) {_max_compute_units=max_compute_units;}
  void set_global_mem_size(size_t global_mem_size) {_global_mem_size=global_mem_size;}
  void set_local_mem_size(size_t local_mem_size) {_local_mem_size=local_mem_size;}
  void set_mode(compute_mode compute_mode){_compute_mode=compute_mode;}
  void set_max_work_group_size(int max_work_group_size) {_max_work_group_size = max_work_group_size;}
  void set_warp_size(int warp_size) { _warp_size = warp_size; }
  void set_max_work_items_per_compute_unit(int max_work_items_per_compute_unit){_max_work_items_per_compute_unit = max_work_items_per_compute_unit;}
  void set_max_grid_size(int max_grid_size[]) {
    for (int i = 0; i < 3; i++)
      _max_grid_size[i] = max_grid_size[i];
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
  int _warp_size;
  int _max_work_items_per_compute_unit;
  size_t _global_mem_size;
  size_t _local_mem_size;
  size_t _max_grid_size[3];
  compute_mode _compute_mode = compute_mode::default_;
};

class syclct_device : public cl::sycl::device {
public:
  syclct_device() : cl::sycl::device() {}
  syclct_device(const cl::sycl::device &base) : cl::sycl::device(base) {
    _default_queue = cl::sycl::queue(base, exception_handler);
  }

  int is_native_atomic_supported() { return 0; }
  int compute_capability_major() { return _major; }
  //....

  void get_device_info(sycl_device_info &out) {
    sycl_device_info prop;
    prop.set_name(get_info<cl::sycl::info::device::name>().c_str());

    // Version string has the following format:
    // OpenCL<space><major.minor><space><vendor-specific-information>
    std::stringstream ver;
    ver << get_info<cl::sycl::info::device::version>();
    std::string item;
    std::getline(ver, item, ' '); // OpenCL
    std::getline(ver, item, '.'); // major
    prop.set_major_version(std::stoi(item));
    _major = std::stoi(item);
    std::getline(ver, item, ' '); // minor
    prop.set_minor_version(std::stoi(item));

    prop.set_max_work_item_sizes(get_info<cl::sycl::info::device::max_work_item_sizes>());
    prop.set_host_unified_memory(get_info<cl::sycl::info::device::host_unified_memory>());
    prop.set_max_clock_frequency(get_info<cl::sycl::info::device::max_clock_frequency>());
    prop.set_max_compute_units(get_info<cl::sycl::info::device::max_compute_units>());
    prop.set_max_work_group_size(get_info<cl::sycl::info::device::max_work_group_size>());
    prop.set_global_mem_size(get_info<cl::sycl::info::device::global_mem_size>());
    prop.set_local_mem_size(get_info<cl::sycl::info::device::local_mem_size>());

    size_t max_sub_group_size = 32;
#ifdef CL_DEVICE_SUB_GROUP_SIZES_INTEL
    cl::sycl::vector_class<size_t> sub_group_sizes = _default_queue.get_device().get_info<cl::sycl::info::device::sub_group_sizes>();
    cl::sycl::vector_class<size_t>::const_iterator max_iter = std::max_element(sub_group_sizes.begin(), sub_group_sizes.end());
    max_sub_group_size = *max_iter;
#endif
    prop.set_warp_size(max_sub_group_size);

    prop.set_max_work_items_per_compute_unit(get_info<cl::sycl::info::device::max_work_group_size>());
    int max_grid_size[] = { 0x7FFFFFFF, 0x7FFFFFFF, 0x7FFFFFFF };
    prop.set_max_grid_size(max_grid_size);
    //...
    out = prop;
  }

  void reset() {
    // release ALL (TODO) resources and reset to initial state
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
  cl::sycl::queue _default_queue;
  std::set<cl::sycl::queue> _queues;
  int _major;
};

class device_manager {
public:
  device_manager() {
    std::vector<cl::sycl::device> sycl_gpu_devs =
        cl::sycl::device::get_devices(cl::sycl::info::device_type::gpu);
    for (auto &dev : sycl_gpu_devs) {
      _devs.push_back(syclct_device(dev));
    }
    std::vector<cl::sycl::device> sycl_cpu_devs =
        cl::sycl::device::get_devices(cl::sycl::info::device_type::cpu);
    for (auto &dev : sycl_cpu_devs) {
      _devs.push_back(syclct_device(dev));
    }
  }
  syclct_device &current_device() {
    check_id(_current_device);
    return _devs[_current_device];
  }
  syclct_device get_device(unsigned int id) const {
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
  std::vector<syclct_device> _devs;
  unsigned int _current_device = 0;
};

static device_manager &get_device_manager() {
  static device_manager d_m;
  return d_m;
}
static inline cl::sycl::queue &get_default_queue() {
  return syclct::get_device_manager().current_device().default_queue();
}

} // namespace syclct

#endif // SYCLCT_DEVICE_H
