To release image:

Insert next tag:
```
docker build -t modulus-core -f Dockerfile.release .. 
```

To build release image with all the Modulus repos:

Create an empty directory and clone all the individual repositories:
```
mkdir build/ build/deps/ && cd build/
git clone https://github.com/NVIDIA/modulus.git
git clone https://github.com/NVIDIA/modulus-launch.git
git clone https://github.com/NVIDIA/modulus-sym.git
```

Download the Optix SDK from https://developer.nvidia.com/designworks/optix/downloads/legacy. Currently Modulus supports v7.0.
Place it in the `deps` directory and make it executable. 
```
chmod +x deps/NVIDIA-OptiX-SDK-7.0.0-linux64.sh 
```

Build the docker image using (ensure you are in the `build` directory)
```
docker build -t modulus-combined:0.1 -f ./modulus/dockerfiles/Dockerfile.release.combined .
```


To build CI image:

Insert next tag based on YY.MM.ID
(E.g. December 2022, 3rd image has tag 22.12.03)
```
docker build -t modulus-ci:xx.xx.xx -f Dockerfile.ci .. 
```
