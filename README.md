![Thumbnail](./assets/thumbnail.png)

Screening of Congenital Heart Diseases (CHD) in mice with 3D <img src="https://latex.codecogs.com/svg.latex?\mu" /> CTscans.

Napari Plugin: [MouseCHD Napari plugin](https://github.com/hnguyentt/mousechd-napari)

## Installation:
* Create virtual environment: `conda create -n mousechd python=3.9`
* Activate the environment: `conda activate mousechd`
* Install the package: `pip install mousechd`
  
## How to use
This 
For user-friendly interface, refer to MouseCHD Napari plugin in this repository: https://github.com/hnguyentt/mousechd-napari

### Raw data structure

It is recommended that your data is structured in the following way:
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

### Preprocessing

This step standardizes the data into the same spacing and view.
Note: although this script supports different file formats, we recommend using NIFFTI format as it contains information about the spacing and orientation of the image.

```bash
mousechd preprocess.py \
    -database <PATH/TO/DATABASE> \
    -maskdir <PATH/TO/MASK/DIR> \ # relative to database
    -masktype NIFFTI \ # relative to database
    -metafile <PATH/TO/META/FILE> \
    -outdir <PATH/TO/OUTPUT/DIR>
```

### Heart segmentation
* Preparing data for training and testing nnUNet:
  ```bash
  mousechd segment -indir [INPUT/DIRECTORY] -outdir [OUTPUT/DIRECTORY]
  ```

* Post-process nnUNet
```bash
mousechd postprocess_nnUNet \
    -indir <PATH/TO/SEGMENTATION/BY/nnUNet> \
    -outdir <PATH/TO/OUTDIR>
```

### CHD detection

## Analysis
For some visualizations (), [Napari] is required. To install: `conda install -c conda-forge napari`
A detailed analysis of the results can be found in the folder [analysis](./analysis/).

## Acknowledgements
* INCEPTION funding: [INCEPTION](https://www.inception-program.fr/en) 
