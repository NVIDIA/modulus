# Introduction

Welcome to Project Modulus! We're excited you're here and want to contribute. This documentation is intended for individuals and institutions interested in contributing to Modulus. Modulus is an open-source project and, as such, its success relies on its community of contributors willing to keep improving it. Your contribution will be a valued addition to the code base; we simply ask that you read this page and understand our contribution process, whether you are a seasoned open-source contributor or whether you are a first-time contributor.

## Communicate with us
We are happy to talk with you about your needs for Modulus and your ideas for contributing to the project. One way to do this is to create an issue discussing your thoughts. It might be that a very similar feature is under development or already exists, so an issue is a great starting point. If you are looking for an issue to resolve that will help, refer to the [issue](https://github.com/NVIDIA/modulus/issues) section.


# Contribute to Modulus-Core

## Pull Requests
Developer workflow for code contributions is as follows:

1. Developers must first [fork](https://help.github.com/en/articles/fork-a-repo) the [upstream](https://github.com/NVIDIA/Modulus) Modulus repository.

2. Git clone the forked repository and push changes to the personal fork.

3. Once the code changes are staged on the fork and ready for review, a [Pull Request](https://help.github.com/en/articles/about-pull-requests) (PR) can be [requested](https://help.github.com/en/articles/creating-a-pull-request) to merge the changes from a branch of the fork into a selected branch of upstream.
  * Exercise caution when selecting the source and target branches for the PR.
  * Creation of a PR creation kicks off CI and a code review process.
  * Atleast one Modulus engineer will be assigned for the review.

4. The PR will be accepted and the corresponding issue closed after adequate review and testing has been completed.

## Licensing information
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

## CI

To ensure quality of the code, your merge request will pass through several CI checks. 
It is mandatory for your MRs to pass these pipelines to ensure a successful merge. 
Please keep checking this document for the latest guidelines on pushing code. Currently, 
The pipeline has following stages:

1. `black` 
  Checks for formatting of your python code. 
  Refer [black](https://black.readthedocs.io/en/stable/) for more information. 
  If your MR fails this test, run `black <script-name>.py` on problematic scripts and 
  black will take care of the rest. 

2. `interrogate` 
  Checks if the code being pushed is well documented. The goal is to make the 
  documentation live inside code. Very few exceptions are made. 
  Elements that are fine to have no documentation include `init-module`, `init-method`, 
  `private` and `semiprivate` classes/functions and `dunder` methods. For definitions of 
  these, refer [interrogate](https://interrogate.readthedocs.io/en/latest/). Meaning for
  some methods/functions is very explicit and exceptions for these are made. These 
  include `forward`, `reset_parameters`, `extra_repr`, `MetaData`. If your MR fails this
  test, add the missing documentation. Take a look at the pipeline output for hints on 
  which functions/classes need documentation. 
  To test the documentation before making a commit, you can run the following during 
  your development  
  ```
  interrogate \
    --ignore-init-method \
    --ignore-init-module \
    --ignore-module \
    --ignore-private \
    --ignore-semiprivate \
    --ignore-magic \
    --fail-under 99 \
    --exclude '[setup.py]' \
    --ignore-regex forward \
    --ignore-regex reset_parameters \
    --ignore-regex extra_repr \
    --ignore-regex MetaData \
    -vv \
    --color \
    ./modulus/
  ```

3. `pytest` 
  Checks if the test scripts from the `test` folder run and produce desired outputs. It 
  is imperative that your changes don't break the existing tests. If your MR fails this
  test, you will have to review your changes and fix the issues. 
  To run pytest locally you can simply run `pytest` inside the `test` folder.

4. `doctest` 
  Checks if the examples in the docstrings run and produce desired outputs. It is highly
  recommended that you provide simple examples of your functions/classes in the code's
  docstring itself. Keep these examples simple and also add the expected outputs. 
  Refer [doctest](https://docs.python.org/3/library/doctest.html) for more information. 
  If your MR fails this test, check your changes and the docstrings. 
  To run doctest locally, you can simply run `pytest --doctest-modules` inside the 
  `modulus` folder. 

5. `coverage`
  Checks if your code additions have sufficient coverage. Refer 
  [coverage](https://coverage.readthedocs.io/en/6.5.0/index.html#) for more details. If 
  your MR fails this test, this means that you have not added enough tests to the `test`
  folder for your module/functions. Add extensive test scripts to cover different 
  branches and lines of your additions. Aim for more than 80% code coverage. 
  To test coverage locally, run the `get_coverage.sh` script from the `test` folder and
  check the coverage of the module that you added/edited. 

## pre-commit

We use `pre-commit` to performing formatting checks before the changes are commited. 

To enable `pre-commit` follow the below steps:

```
pip install pre-commit
pre-commit install
```

Once the above commands are executed, the pre-commit hooks will be activated and all 
the commits will be checked for appropriate formatting.
