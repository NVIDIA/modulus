# Getting Started

## 1. Introduction & Context
This guide describes how to set up a machine for projects involving **NVIDIA Modulus (version 24.12)** on **Ubuntu 22.04 LTS**. It covers:
- Setting up Git (including configuring user info and managing SSH keys).
- Installing NVIDIA drivers, Docker, and the NVIDIA Container Toolkit.
- Pulling and running the Modulus 24.12 container.
- Installing Python dependencies and running Jupyter inside the container.

> **Disclaimer**:  
> - These steps have been tested for **online** installation only.  
> - An **offline (air-gapped)** approach is possible but not thoroughly tested. See the **Appendix** (not included here) for a suggested offline workflow.

---

## 2. Set Up Git

### 2.1 Configure Git User Info

Before cloning repositories, ensure Git is configured with your user details:

```bash
git config --global user.email "user@example.com"
git config --global user.name "YGMaerz"
```

### 2.2 Manage SSH Keys

There are two main approaches: **creating a new key** or **adding an existing key** to this machine.

#### **Option A: Create a New SSH Key**

1. Generate an **ed25519** key pair:
   ```bash
   ssh-keygen -t ed25519 -C "user@example.com" -q -N "" -f ~/.ssh/id_ed25519_key
   ```
2. Retrieve the **public** key:
   ```bash
   cat ~/.ssh/id_ed25519_key.pub
   ```
3. Add this public key to your Git provider (e.g., GitHub).  
4. (Optional) Verify permissions:
   ```bash
   chmod 700 ~/.ssh
   chmod 600 ~/.ssh/id_ed25519_key
   chmod 644 ~/.ssh/id_ed25519_key.pub
   chmod 644 ~/.ssh/known_hosts  # if you have known_hosts entries
   ```

#### **Option B: Add an Existing SSH Key**
If you already have a key pair, simply copy it into `~/.ssh/` on your new machine and ensure correct permissions as above.

### 2.3 Configure SSH for GitHub
Add or update your `~/.ssh/config` (creating the file if it doesn’t exist):

```bash
cat <<EOF >> ~/.ssh/config

Host github.com
  AddKeysToAgent yes
  IdentityFile ~/.ssh/id_ed25519_key
EOF
```

Then set file permissions:
```bash
chmod 700 ~/.ssh
chmod 600 ~/.ssh/config
```

---

## 3. Set Up the Project

1. **Create a project folder** (if you haven’t already):
   ```bash
   cd 
   mkdir project
   ```
2. **Clone the repositories**:
   ```bash
   ssh-keyscan github.com >> ~/.ssh/known_hosts
   cd project
   git clone git@github.com:NVIDIA/modulus.git
   git clone git@github.com:YGMaerz/modulus-dls-api.git
   cd ..
   ```

---

## 4. Set Up the Machine

### 4.1 Install NVIDIA Driver (Online)

Install the NVIDIA 525 driver on Ubuntu 22.04:

```bash
sudo apt-get update
sudo apt-get install -y nvidia-driver-525
sudo reboot
```

After reboot, verify the driver is installed:

```bash
nvidia-smi
```

---

### 4.2 Install Docker

1. **Remove pre-installed Docker packages** (if any):
   ```bash
   for pkg in docker.io docker-doc docker-compose docker-compose-v2 podman-docker containerd runc; do
     sudo apt-get remove -y $pkg
   done
   ```

2. **Add Docker’s official GPG key & repository**:
   ```bash
   sudo apt-get update
   sudo apt-get install -y ca-certificates curl

   # Create a directory for the Docker key if it doesn't exist
   sudo install -m 0755 -d /etc/apt/keyrings
   sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
       -o /etc/apt/keyrings/docker.asc
   sudo chmod a+r /etc/apt/keyrings/docker.asc

   # Add Docker repo to APT sources
   echo \
     "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] \
     https://download.docker.com/linux/ubuntu \
     $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
     sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
   ```

3. **Install Docker packages**:
   ```bash
   sudo apt-get update
   sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
   ```

4. **Enable non-root Docker usage**:
   ```bash
   sudo usermod -aG docker $USER
   sudo reboot
   ```

5. **Verify Docker installation**:
   ```bash
   sudo docker run hello-world
   ```

---

### 4.3 Install NVIDIA Container Toolkit

1. **Add the NVIDIA Container Toolkit Repository**:

   ```bash
   curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
       | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

   curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
       | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
       | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

   sudo apt-get update
   sudo apt-get install -y nvidia-container-toolkit
   ```

2. **(Optional) Install `nvidia-docker2`**:
   ```bash
   distribution=$(. /etc/os-release; echo $ID$VERSION_ID)
   curl -s -L https://nvidia.github.io/libnvidia-container/gpgkey | sudo apt-key add -
   curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list \
       | sudo tee /etc/apt/sources.list.d/nvidia-container-runtime.list

   sudo apt-get update
   sudo apt-get install -y nvidia-docker2
   sudo systemctl restart docker
   ```

---

## 5. Install & Run NVIDIA Modulus (24.12)

### 5.1 Pull Modulus Docker Image

```bash
docker pull nvcr.io/nvidia/modulus/modulus:24.12
```

### 5.2 Launch the Modulus Container

1. **Navigate** to the project folder:

   ```bash
   cd project/modulus-dls-api/
   ```

2. **Run the container** (interactive mode and published port 8888):

   ```bash
   docker run --gpus all \
              --shm-size=1g \
              --ulimit memlock=-1 \
              --ulimit stack=67108864 \
              -v "${PWD}:/workspace" \
              --name my_modulus_container \
              -it \
              -p 8888:8888 \
              nvcr.io/nvidia/modulus/modulus:24.12 bash
   ```

### 5.3 Install Python Requirements
Inside the container, install any project-specific dependencies:
```bash
pip install -r examples/cfd/darcy_autoML_active_learning/requirements.txt
```

### 5.4 Exit Container
- **Leave** the container:
  ```bash
  exit
  ```

---

## 6. Restarting the Container & Running Jupyter Notebook

### 6.1 Restart & Attach to the Container
If you exited your running container, you can easily restart and reattach to it:
```bash
docker start my_modulus_container
docker exec -it my_modulus_container bash
```

### 6.2 (Optional) Launch Jupyter Notebook
1. **Start Jupyter** inside the container:
   ```bash
   jupyter notebook --ip=0.0.0.0 --port=8888 --allow-root --no-browser
   ```

2. **Access Jupyter** on your host:
   - Ensure you **published** port `8888` when first running the container (using `-p 8888:8888`).
   - On your host machine, open `http://localhost:8888` in a browser.  
   - You’ll see the Jupyter Notebook interface, allowing you to create and run notebooks within the container environment.

---

# Appendix: Offline Installation (e.g., for Airgap)

> **Disclaimer**: This approach is not fully tested and depends on your ability to correctly download and transfer all required files. Adjust paths, filenames, and versions as needed.

## 1. Prepare on an Online Machine

### 1.1 Gather Required Packages

1. **Ubuntu packages**  
   You’ll need `.deb` packages for:
   - **NVIDIA Driver** (e.g., 525) if it’s not already installed on your offline machine.  
   - **Docker** (Docker CE, CLI, containerd, etc.).  
   - **NVIDIA Container Toolkit** (e.g., `nvidia-docker2`, `nvidia-container-toolkit`).  

   > **Tip**: On the online machine, configure the same Ubuntu version (22.04) repositories, then do:
   > ```bash
   > apt-get update
   > apt-get download <package_name>
   > ```
   > This will download `.deb` files locally instead of installing them. You can also check official repositories or use apt-cacher methods.

2. **NVIDIA Modulus Docker Image**  
   - Pull the image on the online machine:
     ```bash
     docker pull nvcr.io/nvidia/modulus/modulus:24.12
     ```
   - **Save** the image to a file:
     ```bash
     docker save -o modulus_24.12.tar nvcr.io/nvidia/modulus/modulus:24.12
     ```
   - Later, you’ll transfer `modulus_24.12.tar` to your offline machine and load it into Docker.

3. **Git Repositories or Source Code**  
   - If your offline machine won’t have direct GitHub access, **clone** or **archive** the repositories on the online machine.
     ```bash
     git clone git@github.com:NVIDIA/modulus.git
     git clone git@github.com:YGMaerz/modulus-dls-api.git
     ```
   - You can also compress them:
     ```bash
     tar -czf modulus.tar.gz modulus
     tar -czf modulus-dls-api.tar.gz modulus-dls-api
     ```

4. **Python Dependencies**  
   - If the project has a `requirements.txt` file, you can **download wheels** using:
     ```bash
     pip download -r requirements.txt -d ./offline_wheels
     ```
   - Transfer the entire `offline_wheels` folder to the offline machine.

### 1.2 Transfer Files to Offline Machine

1. **Copy everything** (the `.deb` packages, `.tar` Docker images, zipped repositories, Python wheels, etc.) onto removable media (USB drive, external HDD).
2. **Move** them to your offline machine.

---

## **2. Install on the Offline Machine**

Once you have all required files on the offline machine, follow these steps:

### 2.1 Install Ubuntu Packages from `.deb` Files

1. **NVIDIA Driver**  
   - If your system doesn’t already have the correct driver installed, install the `.deb` package(s) you downloaded:
     ```bash
     sudo dpkg -i nvidia-driver-525_*.deb
     ```
   - Reboot to load the new driver:
     ```bash
     sudo reboot
     ```
   - Verify:
     ```bash
     nvidia-smi
     ```

2. **Docker Engine & Dependencies**  
   - Remove any existing Docker packages (optional but recommended):
     ```bash
     for pkg in docker.io docker-doc docker-compose docker-compose-v2 \
       podman-docker containerd runc; do 
         sudo apt-get remove -y $pkg 
     done
     ```
   - Install `.deb` packages for Docker (e.g., `docker-ce`, `docker-ce-cli`, `containerd.io`, `docker-compose-plugin`, etc.):
     ```bash
     sudo dpkg -i docker-ce_*.deb
     sudo dpkg -i docker-ce-cli_*.deb
     sudo dpkg -i containerd.io_*.deb
     sudo dpkg -i docker-buildx-plugin_*.deb
     sudo dpkg -i docker-compose-plugin_*.deb
     ```
   - (Optional) If there are dependency issues, run:
     ```bash
     sudo apt-get install -f
     ```
   - Add your user to the `docker` group and reboot or re-login:
     ```bash
     sudo usermod -aG docker $USER
     sudo reboot
     ```
   - Test Docker:
     ```bash
     sudo docker run hello-world
     ```

3. **NVIDIA Container Toolkit**  
   - Install `.deb` packages for `nvidia-docker2`, `nvidia-container-toolkit`, or relevant `.deb` files.
     ```bash
     sudo dpkg -i nvidia-container-toolkit_*.deb
     sudo dpkg -i nvidia-docker2_*.deb
     ```
   - Restart Docker so it picks up the new runtime:
     ```bash
     sudo systemctl restart docker
     ```

### 2.2 Load the NVIDIA Modulus Docker Image

1. **Load from Saved Tar**  
   ```bash
   docker load -i modulus_24.12.tar
   ```
   This imports the image `nvcr.io/nvidia/modulus/modulus:24.12` into your local Docker registry.

2. **Verify**  
   ```bash
   docker images
   ```
   You should see `nvcr.io/nvidia/modulus/modulus:24.12` in the list.

### 2.3 Prepare Git Repos / Project Files

If you transferred the repos as `.tar.gz` archives:
```bash
tar -xzf modulus.tar.gz
tar -xzf modulus-dls-api.tar.gz
```
Place them into your desired `project/` directory.

- If you’re using **SSH keys** on the offline machine, ensure you have your `~/.ssh` directory set up with the right permissions:
  ```bash
  chmod 700 ~/.ssh
  chmod 600 ~/.ssh/id_ed25519_key
  chmod 644 ~/.ssh/id_ed25519_key.pub
  chmod 644 ~/.ssh/known_hosts
  ```

### 2.4 Install Python Dependencies

If your project requires Python packages:
1. **Navigate** to the project directory (e.g., `modulus-dls-api`).
2. **Install** the packages from local wheels:
   ```bash
   pip install --no-index --find-links=./offline_wheels -r requirements.txt
   ```
   - `--no-index`: prevents pip from trying to reach PyPI.  
   - `--find-links=./offline_wheels`: points pip to your local folder of wheels.

---

## 3. Run & Verify

1. **Start the Modulus Container**  
   ```bash
   cd project/modulus-dls-api/
   docker run --gpus all \
              --shm-size=1g \
              --ulimit memlock=-1 \
              --ulimit stack=67108864 \
              -v "${PWD}:/workspace" \
              --name my_modulus_container \
              -it \
              nvcr.io/nvidia/modulus/modulus:24.12 bash
   ```
   > **Note**: If you plan to run Jupyter inside the container and need to access it from your host, consider adding `-p 8888:8888`.

2. **Install Additional Python Requirements** (if not done offline in step 2.4):
   ```bash
   pip install -r examples/cfd/darcy_autoML_active_learning/requirements.txt
   ```

3. **Run Jupyter** (if desired):
   ```bash
   jupyter notebook --ip=0.0.0.0 --port=8888 --allow-root --no-browser
   ```
   - Access it via `http://localhost:8888` on your host machine if you published port 8888 in the `docker run` command.

4. **Stop / Restart Container**  
   - **Stop** container from inside:
     ```bash
     exit
     ```
   - **Restart**:
     ```bash
     docker start my_modulus_container
     docker exec -it my_modulus_container bash
     ```

---

## 4. Final Notes

- **File Hashes**: For better security, you may want to verify checksums (e.g., `sha256sum`) of the transferred `.deb` packages, Docker images, and archives.
- **Permissions**: Always confirm SSH folder and file permissions:
  ```bash
  chmod 700 ~/.ssh
  chmod 600 ~/.ssh/id_ed25519_key
  chmod 644 ~/.ssh/id_ed25519_key.pub
  chmod 644 ~/.ssh/known_hosts
  ```
- **Dependencies**: If you encounter **dependency issues** while installing `.deb` packages, run:
  ```bash
  sudo apt-get install -f
  ```
  or manually install the missing dependencies you also downloaded.
- **Updates**: If you need to update or install new packages, repeat the offline download/transfer process with the updated packages.
- **Modulus Versions**: This example references Modulus version **24.12**. Adjust if you need a different version.

---

This **Appendix** should help guide you through setting up your environment **offline**. Make sure you have all required components downloaded and transferred before starting, and verify installation steps carefully at each stage.