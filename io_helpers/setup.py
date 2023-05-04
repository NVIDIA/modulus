# Copyright (c) 2020-2022, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from setuptools import setup, Extension
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(name='io_helpers',
      ext_modules=[CppExtension(name='io_helpers',
                                sources=['cpp/direct_io.cpp',
                                         'cpp/interface.cpp'],
                                extra_compile_args={'cxx': ['-g', '-O2']})
      ],
      cmdclass={'build_ext': BuildExtension})
