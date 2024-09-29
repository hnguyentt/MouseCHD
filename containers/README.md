# Containers for mousechd package

## Docker

### Running docker with GPU
Running docker with GPU requires installing `nvidia-container-toolkit` package. If you haven't had it install, following this instruction to install.

On Debian-based OSes:
```bash
# Add the package repositories
$ distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
$ curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
$ curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

$ sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
$ sudo systemctl restart docker
```
(See: https://stackoverflow.com/a/58432877/11524628) for more details.

### Running `mousechd` with docker
* Pull `mousechd` docker image: `sudo docker pull hoanguyen93/mousechd`
* Append this prefix before every command in the [How to use](https://github.com/hnguyentt/MouseCHD?tab=readme-ov-file#how-to-use): `sudo docker run --gpus all -v /path/on/host:/path/in/container mousechd`
  * `--gpus all`: container can see and use all gpus available on host
  * `-v /path/on/host:/path/in/container`: mount host data to container. For example, I mount my home folder to container home folder: `-v /home/hnguyent:/homme/hnguyent`

For example, I run [segmentation](https://github.com/hnguyentt/MouseCHD?tab=readme-ov-file#2-heart-segmentation) like this: `sudo docker run --gpus all -v /home/hnguyent:/home/hnguyent mousechd mousechd segment -indir /home/hnguyent/DATA/images -outdir /home/hnguyent/DATA/HeartSeg`


## Apptainer
[Apptainer](https://apptainer.org/) is an open-source containerization tool, designed specifically for High-Performance Computing (HPC) environments. Unlike traditional container solutions like Docker, Apptainer emphasizes **unprivileged execution**, ensuring safe deployment in shared computing infrastructures, where running services as root may pose security risks. Apptainer enables portability of applications across different systems, while allowing access to GPUs, specialized hardware, and libraries.

### Downloading models in advance on HPC
On HPC, the internet connection may not be available on running node, you should download models in advance, copy and paste in command line on HPC:

```
ver=13785314
mkdir -p ~/.MouseCHD/Classifier/"$ver" && cd ~/.MouseCHD/Classifier/"$ver" && wget https://zenodo.org/records/"$ver"/files/Classifier.zip && unzip Classifier.zip && rm Classifier.zip
mkdir -p ~/.MouseCHD/HeartSeg/"$ver" && cd ~/.MouseCHD/HeartSeg/"$ver" && wget https://zenodo.org/records/"$ver"/files/HeartSeg.zip && unzip HeartSeg.zip && rm HeartSeg.zip && cd ~
```

### Running `mousechd` with Apptainer
* Download Apptainer: `wget https://zenodo.org/records/13855119/files/mousechd.sif`
* Download model in advance (only on HPC) like the [instruction above]().
* Append this prefix before every command in the [How to use](https://github.com/hnguyentt/MouseCHD?tab=readme-ov-file#how-to-use): `apptainer exec [-B /path/to/directory/on/host] --nv <path/to/mousechd.sif>`:
  * `--nv`: container can see and use available gpus.
  * `-B /path/to/directory/on/host`: `B` flag is for binding directories from the host into the container. If your data is not visible by container, especially on HPC, you could use this flag to mount your data to container.

For example, I run [segmentation](https://github.com/hnguyentt/MouseCHD?tab=readme-ov-file#2-heart-segmentation) like this: 

* On my local machine: `sapptainer exec --nv mousechd.sif mousechd segment -indir /home/hnguyent/DATA/images -outdir /home/hnguyent/DATA/HeartSeg`.

* On HPC, I append additional SLURM prefix to request for computational resources: `srun -J "mousechd" -p gpu --qos=gpu --gres=gpu:1 --cpus-per-task=2 --mem-per-cpu=250000 apptainer exec -B /mount_data --nv mousechd.sif DATA/zeus/hnguyent/images -outdir DATA/zeus/hnguyent/HeartSeg`. Here I need to use `-B` flag to make my data visible in container.
