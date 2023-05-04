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

#include "direct_io.h"

// routine for IO load
py::bytes load_file_direct(const std::string& filename, const size_t& blocksize, size_t filesize) {

  // open file, not with O_DIRECT at the beginning
  int fd = open(filename.c_str(), O_RDONLY);
  if(fd == -1){
    throw std::runtime_error("Error, cannot open file " + filename + ".");
  }

  // get stats
  if (filesize == 0) {
    struct stat sb;
    fstat(fd, &sb);
    filesize = sb.st_size;
  }

  // advise for IO, just to make sure
  posix_fadvise(fd, 0, filesize, POSIX_FADV_SEQUENTIAL | POSIX_FADV_DONTNEED);

  // allocate aligned buffer, multiples of 512B:
  const size_t alignment = static_cast<size_t>(blocksize);
  size_t buffsize = ((filesize + alignment - 1) / alignment) * alignment;
  size_t alignedsize = (filesize / alignment) * alignment;

  //auto buff = static_cast<char*>(std::aligned_alloc(alignment, buffsize));

  auto buff = static_cast<char*>(memalign(alignment, buffsize));
  if (buff == nullptr) {
    throw std::runtime_error("Error allocating IO buffer.");
  }

  // read unaligned portion first:
  auto readsize = filesize - alignedsize;
  off_t off = alignedsize;
  while (readsize > 0) {
    auto nread = pread(fd, buff + off, readsize, off);

    if (nread >= 0) {
      // worked well, continue
      readsize -= nread;
      off += nread;
    }
    else {
      throw std::runtime_error("Error when reading remainder from file " + filename + ".");
    }
  }

  // now set fd to O_DIRECT and tread the aligned portion:
  auto status = fcntl(fd, F_SETFL, O_DIRECT);
  if (status < 0) {
    throw std::runtime_error("Error changing status flags for file " + filename + ".");
  }

  // deal with aligned portion now:
  off = 0;
  readsize = alignedsize;
  while (readsize > 0) {
    auto nread = pread(fd, buff + off, readsize, off);

    if (nread >= 0) {
      // worked well, continue
      readsize -= nread;
      off += nread;
    }
    else {
      throw std::runtime_error("Error when reading bulk from file " + filename + ".");
    }
  }

  // close file
  close(fd);

  // copy to string to convert to python byte:
  std::string token(buff, filesize);

  // deallocate memory
  std::free(buff);

  return py::bytes(token);
}


// routine for IO save
size_t save_file_direct(const std::string& filename, const std::string& data, const size_t& blocksize, const bool& sync) {

  // convert data to str:
  //const std::string data = std::string(bdata);
  
  // open file
  int flags = O_CREAT | O_WRONLY | O_TRUNC | O_DIRECT;
  if (sync) flags |= O_SYNC;
  int fd = open(filename.c_str(), flags, S_IRWXU);
  if(fd == -1){
    throw std::runtime_error("Error, cannot open file " + filename + ".");
  }

  // get stats
  auto filesize = data.size();

  // advise for IO, just to make sure
  posix_fadvise(fd, 0, filesize, POSIX_FADV_SEQUENTIAL | POSIX_FADV_DONTNEED);

  // allocate aligned buffer, multiples of 512B:
  const size_t alignment = static_cast<size_t>(blocksize);
  size_t buffsize = ((filesize + alignment - 1) / alignment) * alignment;
  //size_t alignedsize = (filesize / alignment) * alignment;

  //auto buff = static_cast<char*>(std::aligned_alloc(alignment, buffsize));

  auto buff = static_cast<char*>(memalign(alignment, buffsize));
  if (buff == nullptr) {
    throw std::runtime_error("Error allocating IO buffer.");
  }

  // copy data into output buffer
  std::memcpy(buff, data.c_str(), filesize);

  // write:
  off_t off = 0;
  size_t writesize = buffsize;
  while (writesize > 0) {
    auto nwrite = pwrite(fd, buff + off, writesize, off);

    if (nwrite >= 0) {
      // worked well, continue
      writesize -= nwrite;
      off += nwrite;
    }
    else {
      throw std::runtime_error("Error when writing bulk to file " + filename + ".");
    }
  }

  // truncate file to correct length
  auto status = ftruncate(fd, filesize);
  if(status < 0){
    throw std::runtime_error("Error, truncating file " + filename + " failed.");
  }

  // close file
  close(fd);

  // deallocate memory
  std::free(buff);

  return filesize;
}
