## Contributing examples to Modulus Repo

### Pre-requisites:
1. Read the [code contribution guidelines](https://github.com/NVIDIA/modulus/blob/main/CONTRIBUTING.md)
2. What needs to be part of the submission
    1. Training scripts and corresponding datasets for other community members to replicate
    2. Inference scripts and checkpoints for other community members to replicate
3. Please make sure the code and dataset are cleared to be open-sourced under Apache 2.0
4. Make yourself familiar with Modulus-Core, Modulus-Launch or Modulus-Sym as appropriate:
    1. [Modulus-Core](https://github.com/NVIDIA/modulus) includes a collection of models, utilities, and dataloaders
    2. [Modulus-Launch](https://github.com/NVIDIA/modulus-launch) contains a collection of training recipes and examples. Modulus is a dependency of Modulus-Launch but not vice versa 
    3. [Modulus-Sym](https://github.com/NVIDIA/modulus-sym) provides an abstraction layer for using PDE-based symbolic loss functions.

### Recommended Best practices 
1. Wherever possible, avoid  duplicate code. Example: 
    1. Use the network implementations already in Modulus.
    2. Try using the dataloaders and utilities that already exists in Modulus.
    3. Trying using the existing logging utilities [example](https://github.com/NVIDIA/modulus-launch/blob/04f598c4556eb598630946816f01dd97467621de/examples/cfd/vortex_shedding_mgn/train.py#L188).
2. Please follow the checkpointing convention/utility in Modulus for saving and loading your checkpoints [example](https://github.com/NVIDIA/modulus-launch/blob/04f598c4556eb598630946816f01dd97467621de/examples/cfd/vortex_shedding_mgn/train.py#L180).

### General Instructions for adding examples
1. Organizing your code and artifacts as follows:
    1. Create a new folder with the same name in the appropriate folders
    2. Architecture/dataloaders/utilities go to Modulus-Core:
        - Network architectures: [under models](https://github.com/NVIDIA/modulus/tree/main/modulus/models)
        - Utilities: [under utils](https://github.com/NVIDIA/modulus/tree/main/modulus/utils)
        - Dataloader: [under datapipes](https://github.com/NVIDIA/modulus/tree/main/modulus/datapipes)
    3. Training/validation/inference scripts and the relevant utilities go to Modulus-Launch. Place these scripts in a folder under [examples in Modulus-Launch](https://github.com/NVIDIA/modulus-launch/tree/main/examples).

2. Per the contributing document: 
    1. Make forks of Modulus-Core and Modulus-Launch
    2. Push your code to your forks
    3. Once ready, open a PR to the upstream branches
    4. Make sure you prepare your code according to the CI requirements. That includes unit tests and code coverage, docstrings, black formatting, and doctests.

3. Please move all of your dependencies to Modulus’s pyproject.toml.

4. Please make sure your dependencies and any of the sub-dependencies are not touching GPL or L-GPL code. In rare cases, we might be able to clear a code that touches L-GPL code but there are hard restrictions on GPL code.

5. Please add documentation with all your results and plots as a README.md file in Modulus_launch. Modulus documentation is being refactored and we will migrate your documentation to Modulus documentation once ready.

### Licensing information
All source code files should start with this paragraph:
```
# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
```
### Signing Your Work

* We require that all contributors "sign-off" on their commits. This certifies that the contribution is your original work, or you have rights to submit it under the same license, or a compatible license.

  * Any contribution which contains commits that are not Signed-Off will not be accepted.

* To sign off on a commit you simply use the `--signoff` (or `-s`) option when committing your changes:
  ```bash
  $ git commit -s -m "Add cool feature."
  ```
  This will append the following to your commit message:
  ```
  Signed-off-by: Your Name <your@email.com>
  ```

* Full text of the DCO:

  ```
    Developer Certificate of Origin
    Version 1.1
    
    Copyright (C) 2004, 2006 The Linux Foundation and its contributors.
    1 Letterman Drive
    Suite D4700
    San Francisco, CA, 94129
    
    Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
  ```

  ```
    Developer's Certificate of Origin 1.1
    
    By making a contribution to this project, I certify that:
    
    (a) The contribution was created in whole or in part by me and I have the right to submit it under the open source license indicated in the file; or
    
    (b) The contribution is based upon previous work that, to the best of my knowledge, is covered under an appropriate open source license and I have the right under that license to submit that work with modifications, whether created in whole or in part by me, under the same open source license (unless I am permitted to submit under a different license), as indicated in the file; or
    
    (c) The contribution was provided directly to me by some other person who certified (a), (b) or (c) and I have not modified it.
    
    (d) I understand and agree that this project and the contribution are public and that a record of the contribution (including all personal information I submit with it, including my sign-off) is maintained indefinitely and may be redistributed consistent with this project or the open source license(s) involved.
    
### CI
Coming soon
