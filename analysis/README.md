# Analysis
This folder contain plotting code for Figures in the paper. Some scripts require Napari to run: `pip install "napari[all]"`.

You can also run these notebooks with:
* Apptainer: `apptainer exec --nv <path/to/mousechd.sif> jupyter-notebook` (using `-B` to mount data if necessary)
* Docker: 
  * `sudo docker run --gpus all -v <path/on/host>:</path/on/image> -it -p 8888:8888 hoanguyen93/mousechd`
  * Inside the container: `jupyter notebook --ip 0.0.0.0 --port 8888 --no-browser --allow-root`


## Main figures
### Figure 1
* Figure 1a: see [scripts/09_analyze.sh](../scripts/09_analyze.sh)
  * Plot 3D stages: `mousechd viz3d_stages -h` to see the instruction. Example:
    ```bash
    mousechd viz3d_stages \
    -imdir "$PROCESSDIR/Imagine/images" \
    -maskdir "$PROCESSDIR/Imagine/heart-masks" \
    -stage "E18.5" \
    -imname "NH_1045" \
    -savedir "$SAVEDIR/Fig01" \
    -suffix "bbx" \
    -bbx "both" \
    -zoom 2.8 \
    -cam_angles "(0,30,15)" \
    -crop 0
    ```

  * Plot 2D stages: `mousechd viz_stacks -h` to see the instruction. Example
    ```mousechd viz_stacks \
    -impath "$PROCESSDIR/Imagine/images/NH_221c" \
    -crop "axial" \
    -annotate 1 \
    -maskpath "$PROCESSDIR/Imagine/heart-masks/NH_221c" \
    -pad "(1,1,1)" \
    -num 5 \
    -trans_val 30 \
    -savedir "$SAVEDIR/Fig01" \
    -plot_zline 1 \
    -zline_color 'b' \
    -plot_border 1 \
    -color 'b'
    ```

* Figure 1b & 1c: See [01_eda.ipynb](./01_eda.ipynb)

### Figure 3
* See [02_segmentation.ipynb](./02_segmentation.ipynb)

### Figure 4
* See [03_diagnosis.ipynb](./03_diagnosis.ipynb)
  
### Figure 5
* Figure 5a: [scripts/09_analyze.sh](../scripts/09_analyze.sh)
* Figure 5b-g: [03_diagnosis.ipynb](./03_diagnosis.ipynb)

## Supplementary figures

### Supplementary Figure 1
* See [02_segmentation.ipynb](./02_segmentation.ipynb)

### Supplementary Figure 3
* See [03_diagnosis.ipynb](./03_diagnosis.ipynb)

### Supplementary Figure 4
* See [01_eda.ipynb](./01_eda.ipynb)

### Supplementary Figure 5
* See [scripts/08_explain.sh](../scripts/08_explain.sh)

## Revision
Improve plots from the feedback of reviewers.
See [04.revision.ipynb](./04.revision.ipynb) for more details:

* Improve the readability of Venn diagrams