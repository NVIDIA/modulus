# Contribute to Modulus-Core

To contribute to this repo, simply clone this repo, make your changes/contributions and 
create a merge request. Someone from Modulus core team will review and approve your 
merge request. 

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