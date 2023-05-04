#pragma once

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

#include <torch/extension.h>

#include <fcntl.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <malloc.h>

#include <iostream>
#include <cstdlib>
#include <stdexcept>
#include <vector>
#include <string>
#include <memory>
#include <tuple>
#include <queue>

#include <libaio.h>


typedef std::shared_ptr<iocb> iocb_ptr;

class aio_handler {
 public:
  // constructor type
  aio_handler(const size_t& max_events);
  ~aio_handler();

  // submit IO
  void submit_save_file_direct(const::std::string& filename, const std::string& data, const size_t& blocksize);

  // wait for IO
  void wait_all_save_file_direct();
  
 protected:
  size_t _max_events;
  std::vector<io_event> _events;
  io_context_t _write_ctx;
  io_context_t _read_ctx;
  std::queue<iocb_ptr> _write_queue;
};
