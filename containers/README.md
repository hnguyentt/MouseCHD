# Containers for mousechd package

## Docker

Running docker with GPU requires installing `nvidia-container-toolkit` package. To install on Debian-based OSes:
```bash
# Add the package repositories
$ distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
$ curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
$ curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

$ sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
$ sudo systemctl restart docker
```
(See: https://stackoverflow.com/a/58432877/11524628) for more details.

## Apptainer
[Apptainer](https://apptainer.org/) is an open-source containerization tool, designed specifically for High-Performance Computing (HPC) environments. Unlike traditional container solutions like Docker, Apptainer emphasizes **unprivileged execution**, ensuring safe deployment in shared computing infrastructures, where running services as root may pose security risks. Apptainer enables portability of applications across different systems, while allowing access to GPUs, specialized hardware, and libraries.

1. Build Apptainer image
```bash
sudo apptainer build mousechd.sif mousechd.def
```

2. Interact with mousechd

Append the prefix: `apptainer shell --nv mousechd.sif `