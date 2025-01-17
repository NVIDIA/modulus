# **Locating the `darcy_autoML_active_learning` Directory: Detailed Scenarios**

## Context
This document provides an overview of how we locate the `darcy_autoML_active_learning` directory and its subfolders across different environments (Docker vs. local usage). It describes our scenario-based approach in `path_utils.py` so the paths remain consistent regardless of how or where the code is run.

We have a Python-based project that includes a folder named `darcy_autoML_active_learning`. Within this folder, we need to locate:

1. **`darcy_project_root`** (the `darcy_autoML_active_learning` directory itself)  
2. **`config_file`** (typically `darcy_project_root/config/config.yaml`)  
3. **`data_dir`** (`darcy_project_root/data`)  
4. **`results_dir`** (`darcy_project_root/results`)

Additionally, we maintain a **`repo_root`** concept, which often represents the **root** of the entire repository (`modulus-dls-api/`).

We face variations in environment (Docker vs. non-Docker), variations in how Docker is configured (workspace pointing directly to `modulus-dls-api/` or one directory above it), and whether or not `PROJECT_ROOT` is set. Our goal is to consistently identify the correct paths in all scenarios.

---

## Overview

In our `modulus-dls-api` repository, we rely on a subfolder named `darcy_autoML_active_learning` for crucial project components:

- **`darcy_project_root`**: The `darcy_autoML_active_learning` directory itself.
- **`config_file`**: Stored under `darcy_project_root/config/config.yaml`.
- **`data_dir`**: `darcy_project_root/data`.
- **`results_dir`**: `darcy_project_root/results`.

Additionally, there is a concept of **`repo_root`**, which often represents the top-level repository directory (e.g., `modulus-dls-api/`). We must unify how these paths are identified under a variety of conditions, including:

- Docker vs. local environments.
- Different Docker workspace directories (`modulus-dls-api/` vs. one level above).
- Presence or absence of the `PROJECT_ROOT` environment variable in Docker.
- Local Jupyter notebooks started in various directories.

---

## **1. Project Structure (Illustrative)**

A simplified view of the repository might look like this:

```
modulus-dls-api/
├─ examples/
│  └─ cfd/
│     └─ darcy_autoML_active_learning/
│        ├─ config/
│        │  └─ config.yaml
│        ├─ data/
│        ├─ results/
│        ├─ notebooks/
│        │  └─ darcy_active_learning.ipynb
│        └─ src/
│           └─ darcy_automl_active_learning/
│              └─ path_utils.py
└─ ...
```

---

## **2. General Requirements**

1. **Docker Usage**  
   - We may (or may not) have `PROJECT_ROOT` set as an environment variable.
   - The “workspace” (the Docker working directory) could be either exactly `modulus-dls-api/` or one directory above it, such as `/home/user/`.
   - When the workspace is **exactly** `modulus-dls-api/`, we want to treat it as `"."`.
   
2. **Local (Non-Docker) Usage**  
   - We do **not** rely on `PROJECT_ROOT`.
   - We assume the Jupyter Notebook server can be launched from anywhere (the top-level repo directory or deeper inside).
   - We do **not** trust the current working directory to always be stable. Instead, we rely on obtaining the absolute path of a Python file (like `path_utils.py`) and navigating from there.

3. **Desired Path Forms**  
   - In Docker (when the workspace is the repo root), we prefer to treat that directory as `"."` so that subdirectories appear as `"./examples"`, `"./data"`, etc.
   - In local usage, we also prefer relative paths if possible—but we’ll figure them out by code that references `path_utils.py`.

---

## **3. Enumerated Scenarios**

Below are **eight** scenarios, reflecting Docker vs. non-Docker, plus the presence or absence of `PROJECT_ROOT`, plus the two workspace configurations in Docker.

### **Docker: Workspace = `modulus-dls-api/`**

1. **Scenario A1**: Docker, **workspace** = `modulus-dls-api/`, **`PROJECT_ROOT` is set**  
2. **Scenario A2**: Docker, **workspace** = `modulus-dls-api/`, **`PROJECT_ROOT` is not set**  

### **Docker: Workspace = one directory above** (e.g. `/home/user/`)

3. **Scenario B1**: Docker, **workspace** = one level above `modulus-dls-api/`, **`PROJECT_ROOT` is set**  
4. **Scenario B2**: Docker, **workspace** = one level above `modulus-dls-api/`, **`PROJECT_ROOT` is not set**

### **Local Usage (No Docker)**

Here, we assume `PROJECT_ROOT` is **never** set.

5. **Scenario C1**: Local usage, Jupyter is started in `~/project/modulus-dls-api/`  
6. **Scenario C2**: Local usage, Jupyter is started in `~/project/modulus-dls-api/examples/cfd/darcy_autoML_active_learning/`  
   - We might further vary how many directories we are above or below the top-level. For simplicity, we just illustrate these two.

*(You may or may not need to further expand local usage scenarios, but these are the main ones we foresee.)*

---

## **4. Desired Paths in Each Scenario**

Our code must reliably return the following **five** paths:

1. **`repo_root`**  
   - Often `"."` if the Docker workspace matches the top-level repo or if the user is already in `modulus-dls-api/`.
   - Could be `"./modulus-dls-api"` if the workspace is one directory above.

2. **`darcy_project_root`**  
   - Typically `repo_root/examples/cfd/darcy_autoML_active_learning`.

3. **`config_file`**  
   - Typically `darcy_project_root/config/config.yaml`.

4. **`data_dir`**  
   - `darcy_project_root/data`.

5. **`results_dir`**  
   - `darcy_project_root/results`.

Regardless of environment, these paths should always point to the correct directories/files for the `darcy_autoML_active_learning` project.

This section contains **one table per scenario**, listing how each path (`repo_root`, `darcy_project_root`, `config_file`, `data_dir`, `results_dir`) should look in code.

### **Scenario A1**: Docker, Workspace = `modulus-dls-api/`, `PROJECT_ROOT` **is set**

Even though `PROJECT_ROOT` is set, we consider the workspace (`modulus-dls-api/`) as `"."`. Therefore, **we want**:

| **Path Variable**     | **Desired Value**                                                                              |
|-----------------------|-----------------------------------------------------------------------------------------------|
| `repo_root`           | `.`                                                                                           |
| `darcy_project_root`  | `./examples/cfd/darcy_autoML_active_learning`                                                |
| `config_file`         | `./examples/cfd/darcy_autoML_active_learning/config/config.yaml`                             |
| `data_dir`            | `./examples/cfd/darcy_autoML_active_learning/data`                                           |
| `results_dir`         | `./examples/cfd/darcy_autoML_active_learning/results`                                        |

*(Note: If you **do** want the code to reflect the environment variable’s absolute path, you’d see something like `/workspace/modulus-dls-api`. But you explicitly stated you prefer `.`. This implies code that normalizes or collapses the absolute path to `"."` if it matches the workspace.)*

### **Scenario A2**: Docker, Workspace = `modulus-dls-api/`, `PROJECT_ROOT` **is not set**

Now there is no environment variable. The code sees it’s in `modulus-dls-api/`. We want:

| **Path Variable**     | **Desired Value**                                                                              |
|-----------------------|-----------------------------------------------------------------------------------------------|
| `repo_root`           | `.`                                                                                           |
| `darcy_project_root`  | `./examples/cfd/darcy_autoML_active_learning`                                                |
| `config_file`         | `./examples/cfd/darcy_autoML_active_learning/config/config.yaml`                             |
| `data_dir`            | `./examples/cfd/darcy_autoML_active_learning/data`                                           |
| `results_dir`         | `./examples/cfd/darcy_autoML_active_learning/results`                                        |

### **Scenario B1**: Docker, Workspace = one level above (e.g., `/home/user/`), `PROJECT_ROOT` **is set**

Now, `PROJECT_ROOT` might be `"/home/user/project/modulus-dls-api"` or similar. The code can detect that path, but how do we **want** the final variables to look?

Assume we want them to be **relative** to `project/modulus-dls-api/`, but still displayed in a “nice” manner. Possibly:

| **Path Variable**     | **Desired Value**                                                                              |
|-----------------------|-----------------------------------------------------------------------------------------------|
| `repo_root`           | `./project/modulus-dls-api` (or maybe still `.` if we want to pretend we’re in `modulus-dls-api`) |
| `darcy_project_root`  | `./project/modulus-dls-api/examples/cfd/darcy_autoML_active_learning`                         |
| `config_file`         | `./project/modulus-dls-api/examples/cfd/darcy_autoML_active_learning/config/config.yaml`      |
| `data_dir`            | `./project/modulus-dls-api/examples/cfd/darcy_autoML_active_learning/data`                    |
| `results_dir`         | `./project/modulus-dls-api/examples/cfd/darcy_autoML_active_learning/results`                 |

If you actually want to **collapse** it to `.` meaning `repo_root` is literally `.` from the perspective of Docker, that implies the code detects `"/home/user"` as the workspace but sees `PROJECT_ROOT = "/home/user/project/modulus-dls-api"` and then normalizes it. It’s up to the final design.

### **Scenario B2**: Docker, Workspace = one level above, `PROJECT_ROOT` **is not set**

No environment variable is set, but the user is currently at `/home/user/`. If we rely on the fallback “current working directory is `.`,” then:

| **Path Variable**     | **Desired Value**                                                                                                 |
|-----------------------|------------------------------------------------------------------------------------------------------------------|
| `repo_root`           | `.` (which is `/home/user` in reality)                                                                     |
| `darcy_project_root`  | (Potentially) `./modulus-dls-api/examples/cfd/darcy_autoML_active_learning`                                      |
| `config_file`         | `./modulus-dls-api/examples/cfd/darcy_autoML_active_learning/config/config.yaml`                                 |
| `data_dir`            | `./modulus-dls-api/examples/cfd/darcy_autoML_active_learning/data`                                               |
| `results_dir`         | `./modulus-dls-api/examples/cfd/darcy_autoML_active_learning/results`                                            |

*(We might prefer an error in this scenario if we think it’s invalid for the user to be in a directory above the repo without `PROJECT_ROOT`. Or we might let it proceed with a relative path that includes `modulus-dls-api/` as a subfolder. This is part of the final design to be discussed.)*

### **Scenario C1**: Local, Jupyter started in `~/project/modulus-dls-api/` (no Docker, no `PROJECT_ROOT`)

We do **not** rely on environment variables. We discover `.` is the top-level repo. We prefer:

| **Path Variable**     | **Desired Value**                                                                              |
|-----------------------|-----------------------------------------------------------------------------------------------|
| `repo_root`           | `.`                                                                                           |
| `darcy_project_root`  | `./examples/cfd/darcy_autoML_active_learning`                                                |
| `config_file`         | `./examples/cfd/darcy_autoML_active_learning/config/config.yaml`                             |
| `data_dir`            | `./examples/cfd/darcy_autoML_active_learning/data`                                           |
| `results_dir`         | `./examples/cfd/darcy_autoML_active_learning/results`                                        |

### **Scenario C2**: Local, Jupyter started in `~/project/modulus-dls-api/examples/cfd/darcy_autoML_active_learning/`

We are already inside `darcy_autoML_active_learning`. We might choose:

| **Path Variable**     | **Desired Value**                 |
|-----------------------|-----------------------------------|
| `repo_root`           | `..` (meaning “one directory up”) or maybe you want to call it `../../..` if you define the top-level differently |
| `darcy_project_root`  | `.` (since we are already in `darcy_autoML_active_learning`) |
| `config_file`         | `./config/config.yaml`            |
| `data_dir`            | `./data`                          |
| `results_dir`         | `./results`                       |

*(This depends on whether you define “repo root” as the top-level `modulus-dls-api/` or if you define `darcy_autoML_active_learning` itself as a “root.” This is part of the final design decision. If you do consider `modulus-dls-api` the real root, then `repo_root` might be `../../..`. If you consider the `darcy_autoML_active_learning` folder to be the root of the sub-project, then `.` is your root. Either approach can work.)*

---

## 5. Implementation Approach

We use a **scenario-based** method in `path_utils.py`:

1. **Identify Scenario**:  
   - `identify_scenario()` checks whether we are in Docker, whether `PROJECT_ROOT` is set, and (optionally) the Docker “workspace” location. It returns a code like `"A1"` or `"B2"`.

2. **Scenario-Specific Functions**:  
   - For each recognized scenario (e.g., `get_paths_for_A1()`), we specify exactly how `repo_root` is determined and how subpaths (`darcy_project_root`, `config_file`, etc.) are built.

3. **Public Entry Point**:  
   - `get_paths()` calls `identify_scenario()` and dispatches to the matching function. It returns a tuple (or dictionary) containing `(repo_root, darcy_project_root, config_file, data_dir, results_dir)`.

4. **Fallback / Not Implemented**:  
   - If a scenario is not yet implemented, we raise `NotImplementedError`.

This structure keeps code straightforward: each scenario is in its own function, and the `identify_scenario()` function is small and focused.