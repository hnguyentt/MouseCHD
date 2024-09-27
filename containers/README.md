# Containers for mousechd package

## Docker


## Apptainer
[Apptainer](https://apptainer.org/) is an open-source containerization tool, designed specifically for High-Performance Computing (HPC) environments. Unlike traditional container solutions like Docker, Apptainer emphasizes **unprivileged execution**, ensuring safe deployment in shared computing infrastructures, where running services as root may pose security risks. Apptainer enables portability of applications across different systems, while allowing access to GPUs, specialized hardware, and libraries.

1. Build Apptainer image
```bash
sudo apptainer build mousechd.sif mousechd.def
```

1. Run mousechd

Append the prefix: `apptainer exec --nv mousechd.sif `