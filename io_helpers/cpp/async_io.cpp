// Copyright (c) 2020-2022, NVIDIA CORPORATION. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.    

#include "async_io.h"

aio_handler::aio_handler(const size_t& max_events) : _max_events(max_events) {
  // init write context:
  memset(&_write_ctx, 0, sizeof(_write_ctx));
  auto status = io_setup(_max_events, &_write_ctx);
  if (status < 0) {
    throw std::runtime_error("Error, failed to initialize write context.");
  }

  // init read context
  memset(&_read_ctx, 0, sizeof(_read_ctx));
  status = io_setup(_max_events, &_read_ctx);
  if (status < 0) {
    throw std::runtime_error("Error, failed to initialize read context.");
  }

  // create events list:
  _events.resize(_max_events);
}


aio_handler::~aio_handler() {
  // free contexts
  io_destroy(_write_ctx);
  io_destroy(_read_ctx);
}


void aio_handler::submit_save_file_direct(const::std::string& filename, const std::string& data, const size_t& blocksize) {

  // open file
  int fd = open(filename.c_str(), O_CREAT | O_WRONLY | O_TRUNC | O_DIRECT, S_IRWXU);
  if(fd == -1){
    throw std::runtime_error("Error, cannot open file " + filename + ".");
  }

  // get stats
  auto filesize = data.size();

  // advise for IO, just to make sure
  posix_fadvise(fd, 0, filesize, POSIX_FADV_SEQUENTIAL | POSIX_FADV_DONTNEED);

  // allocate aligned buffer, multiples of blocksize:
  const size_t alignment = static_cast<size_t>(blocksize);
  size_t buffsize = ((filesize + alignment - 1) / alignment) * alignment;
  
  auto buff = static_cast<char*>(memalign(alignment, buffsize));
  if (buff == nullptr) {
    throw std::runtime_error("Error allocating IO buffer.");
  }

  // copy data into output buffer
  std::memcpy(buff, data.c_str(), filesize);
  
  // create io control block
  auto cbp = std::make_shared<iocb>();
  io_prep_pwrite(cbp.get(), fd, buff, filesize, 0);
  cbp.get()->data = reinterpret_cast<void*>(new std::string(filename));
  
  // submio IO:
  struct iocb* cbs[1] = {cbp.get()};
  auto status = io_submit(_write_ctx, 1, cbs);
  if (status < 0) {
    throw std::runtime_error("Error submitting write for file " + filename + ".");
  }

  // add object to code
  _write_queue.push(cbp);
}


void aio_handler::wait_all_save_file_direct() {
  //auto status = io_getevents(_write_ctx, _max_events, max_events, _events.data(), nullptr);
  //if (status < 0) {
  //  throw std::runtime_error("Error, event retrieval failed.");
  //}

  while(!_write_queue.empty()) {
    // retrieve first element from queue
    auto cbpm = _write_queue.front();
    _write_queue.pop();

    // extract info
    auto fd = cbpm.get()->aio_fildes;
    auto filename = *reinterpret_cast<std::string*>(cbpm.get()->data);
    //iocb cb;
    //auto status = io_poll(_write_ctx, &cb, io_callback_t cb, fd, int events);
    //if (status < 0) {
    //  throw std::runtime_error("Error submitting polling for file " + filename + ".");
    //}
  }
}
