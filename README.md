![Thumbnail](https://raw.githubusercontent.com/hnguyentt/MouseCHD/master/assets/thumbnail.png)

Screening of Congenital Heart Diseases (CHD) in mice with 3D <img src="https://latex.codecogs.com/svg.latex?\mu" /> CTscans.

***Napari plugin: [MouseCHD Napari plugin](https://github.com/hnguyentt/mousechd-napari)***


## Installation
There are three ways that you can run the package:

### Conda environment
* Create virtual environment: `conda create -n mousechd python=3.9`
* Activate the environment: `conda activate mousechd`
* Install the package: `pip install mousechd`

### Docker
* Pull the docker image: `sudo docker pull hoanguyen93/mousechd`
* Test if docker image pulled successfully: `sudo docker run mousechd mousechd -h`

<details>
<summary>Expected output:</summary>

```
usage: mousechd [-h] [-version] {postprocess_nnUNet,prepare_nnUNet_data,preprocess,segment,resample,split_data,viz3d_views,viz3d_stages,viz_stacks,viz_eda,viz3d_seg,create_label_df,test_clf,train_clf,explain,viz_grad} ...

optional arguments:
  -h, --help            show this help message and exit
  -version              show program's version number and exit

Choose a command:
  {postprocess_nnUNet,prepare_nnUNet_data,preprocess,segment,resample,split_data,viz3d_views,viz3d_stages,viz_stacks,viz_eda,viz3d_seg,create_label_df,test_clf,train_clf,explain,viz_grad}
```

</details>

To assure that you can run the docker with GPUs if available, see the instruction here: https://github.com/hnguyentt/MouseCHD/tree/master/containers#docker

### Apptainer
In case you run the package on HPC on which you don't have superuser permission, you can use Apptainer instead of docker.

* Download container to your computer or HPC:
```bash
wget https://zenodo.org/records/13850904/files/mousechd.sif
```
* Download models in advance (only on HPC), copy and paste in command line on HPC:
```
ver=13785314
mkdir -p ~/.MouseCHD/Classifier/"$ver" && cd ~/.MouseCHD/Classifier/"$ver"
wget https://zenodo.org/records/"$ver"/files/Classifier.zip
unzip Classifier.zip && rm Classifier.zip
mkdir -p ~/.MouseCHD/HeartSeg/"$ver" && cd ~/.MouseCHD/HearSeg/"$ver"
wget https://zenodo.org/records/"$ver"/files/HeartSeg.zip
unzip HeartSeg.zip && rm HeartSeg.zip && cd ~
```
* Test if container run correctly: `apptainer exec --nv <path/to/mousechd.sif> mousechd -h`

<details>
<summary>Expected output:</summary>

```
usage: mousechd [-h] [-version] {postprocess_nnUNet,prepare_nnUNet_data,preprocess,segment,resample,split_data,viz3d_views,viz3d_stages,viz_stacks,viz_eda,viz3d_seg,create_label_df,test_clf,train_clf,explain,viz_grad} ...

optional arguments:
  -h, --help            show this help message and exit
  -version              show program's version number and exit

Choose a command:
  {postprocess_nnUNet,prepare_nnUNet_data,preprocess,segment,resample,split_data,viz3d_views,viz3d_stages,viz_stacks,viz_eda,viz3d_seg,create_label_df,test_clf,train_clf,explain,viz_grad}
```

</details>
  
## How to use

It is recommended that your data are structured in the following way:
```
    DATABASE # your database name
    └── raw # raw folder to store raw data
        ├── NameOfDataset1 # name of dataset
        │   ├── images_20200206 # folder to store images recieved on 20200206 [YYYYMMDD]
        │   ├── masks_20210115 # folder to store masks recieved on 20210115 [YYYYMMDD]
        │   ├── masks_20210708 # folder to store masks recieved on 20210708 [YYYYMMDD]
        │   └── metadata_20210703.csv # metadata file received on 20210703 [YYYYMMDD]
        └── NameOfDataset2 # name of another dataset
            └── images_20201010
            ......
```

In case you use container, append these prefixes to the following commands:
* Docker: `sudo docker run --gpus all -v /path/on/host:/path/in/container mousechd`
  * `--gpus all`: container can see and use all gpus available on host
  * `-v /path/on/host:/path/in/container`: mount host data to container. For example, I mount my home folder to container home folder: `-v /home/hnguyent:/homme/hnguyent`
* Apptainer: `apptainer exec --nv <path/to/mousechd.sif>`
  * `--nv`: container can see and use available gpus.

### (1) Preprocessing

This step standardizes the data into the same spacing and view.
* Data format supported: "DICOM", "NRRD", "NIFTI"
* Mask data format supported: "TIF2d", "TIF3d", "NIFTI"

```bash
mousechd preprocess \
    -database <PATH/TO/DATABASE> \
    -imdir <PATH/TO/IMAGE/DIR> \ # relative to databse
    -maskdir <PATH/TO/MASK/DIR> \ # relative to database
    -masktype NIFTI \
    -metafile <PATH/TO/META/FILE> \ # csv file with headers: "heart_name", "Stage", "Normal heart", "CHD1", "CHD2", ...
    -outdir "DATA/processed"
```

### (2) Heart segmentation

  ```bash
  mousechd segment -indir "DATA/processed/images" -outdir "OUTPUTS/HeartSeg"
  ```
If your computer crashes when running this, you can decrease the number of threads for preprocessing (`default: 6`)and saving NIFTI files
### (3) CHD detection
```bash
mousechd test_clf \
    -imdir "DATA/processed/images" \
    -maskdir  "OUTPUTS/HeartSeg" \
    -stage ["eval"|"test"] \
    -label [PATH/TO/CSV/TEST/FILE] \ # <optional> if stage is "eval", -label must be specified
    -outdir [PATH/TO/OUTPUT/DIRECTORY]
```

## Retraining

You have the option to retrain the model using your custom dataset. After completing the heart segmentation, resample to augment the data, followed by data splitting and subsequence model retraining.

<details>
<summary>Click here to expand the instruction</summary>

### (1) Resample
```bash
mousechd resample \
    -imdir  "DATA/processed/images" \
    -maskdir  "OUTPUTS/HeartSeg" \
    -outdir "DATA/resampled" \
    -metafile  "DATA/processed/metadata.csv" \
    -save_images 1
```

### (2) Split data
```bash
mousechd split_data \
    -metafile "DATA/processed/metadata.csv" \
    -outdir "DATA/label" \
    -val_size 0.2
```

### (3) Train
```bash
mousechd train_clf \
    -exp_dir "OUTPUTS/Classifier" \
    -exp [EXPERIEMENT_NAME] \
    -data_dir "DATA/resampled" \
    -label_dir "DATA/label/x5_base/1fold" \
    -epochs [NUM_EPOCHS]
```

### (4) Evaluate retrained model
```bash
mousechd test_clf \
    -model_dir "OUTPUTS/Classifier/<EXPERIMENT_NAME>" \
    -imdir "DATA/processed/images" \
    -maskdir  "OUTPUTS/HeartSeg" \
    -stage ["eval"|"test"] \
    -label [PATH/TO/CSV/TEST/FILE] \ # <optional> if stage is "eval", -label must be specified
    -outdir [PATH/TO/OUTPUT/DIRECTORY]
```

</details>

## GradCAM
```bash
mousechd explain \
-exp_dir "OUTPUTS/Classifier/<EXPERIMENT_NAME>" \
-imdir "DATA/resampled/images" \
-outdir [PATH/TO/OUTPUT/DIRECTORY]
```

## Analysis
A detailed analysis can be found in the folder [analysis](./analysis/).

For some visualization, [Napari](https://napari.org/stable/) is required. To install: `pip install "napari[all]"`.


## Acknowledgements
* INCEPTION funding: [INCEPTION](https://www.inception-program.fr/en) 
* GPU server technical support: Quang Tru Huynh
