![Thumbnail](https://raw.githubusercontent.com/hnguyentt/MouseCHD/master/assets/thumbnail.png)

Screening of Congenital Heart Diseases (CHD) in mice with 3D <img src="https://latex.codecogs.com/svg.latex?\mu" /> CTscans.
***Napari plugin: [MouseCHD Napari plugin](https://github.com/hnguyentt/mousechd-napari)***


## Installation
* Create virtual environment: `conda create -n mousechd python=3.9`
* Activate the environment: `conda activate mousechd`
* Install the package: `pip install mousechd`
  
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

### (1) Preprocessing

This step standardizes the data into the same spacing and view.
* Data format supported: "DICOM", "NRRD", "NIFTI"
* Mask data format supported: "TIF2d", "TIF3d", "NIFTI"

```bash
mousechd preprocess.py \
    -database <PATH/TO/DATABASE> \
    -maskdir <PATH/TO/MASK/DIR> \
    -masktype NIFTI \
    -metafile <PATH/TO/META/FILE> \ # csv file with headers: "heart_name", "Stage", "Normal heart", "CHD1", "CHD2", ...
    -outdir "DATA/processed"
```

### (2) Heart segmentation

  ```bash
  mousechd segment -indir "DATA/processed/images" -outdir "OUTPUTS/HeartSeg"
  ```

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
For some visualization, [Napari](https://napari.org/stable/) is required. To install: `pip install "napari[all]`.


## Acknowledgements
* INCEPTION funding: [INCEPTION](https://www.inception-program.fr/en) 
* GPU server technical support: Quang Tru Huynh
