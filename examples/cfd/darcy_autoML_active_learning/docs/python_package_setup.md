# Darcy AutoML & Active Learning: Python Package Setup

This document explains how to set up and install your **Darcy autoML & active learning** code as a **pip-installable Python package**, ensuring smooth imports in both Docker and local environments.

---

## 1. Overview

Historically, the code for Darcy PDE, FNO-based autoML, and active learning was mixed into notebooks with relative imports or `sys.path` hacks. We now transition to a **proper Python package** located under `src/`. This allows:

- **Clean imports** in notebooks: `import darcy_automl_active_learning`
- **Editable installs** (`pip install -e .`), so code changes reflect immediately
- Compatibility with both **Docker** (via your `GETTING_STARTED.md` steps) and **local** usage

---

## 2. Directory Structure

Below is an example folder layout under `modulus-dls-api/examples/cfd/darcy_autoML_active_learning/`:

```
darcy_autoML_active_learning/
├─ notebooks/
│   ├─ darcy_autoML.ipynb
│   ├─ darcy_active_learning.ipynb
├─ src/
│   └─ darcy_automl_active_learning/
│       ├─ __init__.py
│       ├─ AutoMLCandidateModelSelection.py
│       ├─ data_desc_logic.py
│       ├─ ModelTrainingLoop.py
│       ├─ ...
├─ config/
│   └─ config.yaml
├─ pyproject.toml
├─ requirements.txt
└─ README.md
```

### Key Points

1. **Source code** goes in `src/darcy_automl_active_learning/`.  
2. **`pyproject.toml`** defines the package name, version, and dependencies.  
3. **Notebooks** now do imports like `from darcy_automl_active_learning import data_desc_logic` instead of relative path fiddling.

---

## 3. Minimal `pyproject.toml`

Below is an illustrative example:

```toml
[build-system]
requires = ["setuptools>=60.2.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "darcy-automl-active-learning"
version = "0.1.0"
description = "Darcy PDE example with FNO-based autoML and active learning"
authors = [{ name = "YourName" }]
readme = "README.md"
license = { text = "Apache-2.0" }
dependencies = [
    "optuna==4.1.0",
    "mlflow>=2.1.1",
    "tqdm>=4.66.5"
]
# or any other core Python deps

[tool.setuptools.packages.find]
where = ["src"]
```

#### Explanation

- **`where = ["src"]`**: Tells setuptools to find packages under `src/`.
- **`dependencies = [...]`**: List your core Python dependencies (originally in `requirements.txt`). You can keep them pinned or flexible.
- **`name = "darcy-automl-active-learning"`**: The package’s name as PyPI would see it (not strictly used locally, but important to identify the package).

---

## 4. Installing the Package

### 4.1 Docker or Local

From `examples/cfd/darcy_autoML_active_learning/`, run:

```bash
pip install -e .
```

**`-e .`** (editable mode) installs a link to your local `src/darcy_automl_active_learning` code. If you edit `.py` files, those changes appear the next time you run or reload code—no need to reinstall.

### 4.2 Verification

```python
import darcy_automl_active_learning
print(darcy_automl_active_learning.__version__)
```

If the import and version print work, your environment is properly set up.  

No more `ModuleNotFoundError` or `sys.path` tweaks!

---

## 5. Usage in Jupyter Notebooks

Once installed, your notebooks can simply do:

```python
from darcy_automl_active_learning.AutoMLCandidateModelSelection import (
    automl_candidate_model_selection,
    save_candidate_models
)

candidates = automl_candidate_model_selection(...)
```

Or:

```python
from darcy_automl_active_learning import ModelTrainingLoop

ModelTrainingLoop.run_modulus_training_loop(cfg, model, ...)
```

No local path hacks needed—Python sees your package like any other installed module.

---

## 6. HPC / Docker Workflow

1. **Mount or clone** your repository so Docker or HPC sees the `darcy_autoML_active_learning` folder.  
2. **`pip install -e .`** from that folder.  
3. **Run** your notebooks or scripts. They do `import darcy_automl_active_learning.*` without issues.

If using Docker, you can integrate this step into your `Dockerfile` or do it manually after starting the container.

---

## 7. Rationale & Benefits

1. **Cleaner Imports**: No need for `sys.path.append(...)`.  
2. **Editable Installs**: Continuous development—edits in `.py` files reflect immediately.  
3. **Package Organization**: Encourages splitting large notebooks into reusable modules (training loops, data logic, etc.).  
4. **Easier Distribution**: You could upload this package to a private PyPI or share a wheel if needed.

---

## 8. Common Issues / FAQ

### **ModuleNotFoundError**

- **Cause**: You forgot to run `pip install -e .`, or your directory name/package name changed.  
- **Fix**: Verify `pyproject.toml`’s `[project] name = ...` and `where = ["src"]`, then reinstall with `-e`.

### **Edits Not Reflecting**

- **Cause**: You installed normally (not in editable mode), or Jupyter is caching.  
- **Fix**: Ensure you used `-e .`. In Jupyter, you can `%load_ext autoreload` and `%autoreload 2` for dynamic reloading of modules.

### **Dependency Conflicts**

- **Cause**: You pinned dependencies in `pyproject.toml` that conflict with an existing environment.  
- **Fix**: Either loosen version pins, or manage them in a separate environment. For advanced usage, consider tools like `poetry` or `conda`.
