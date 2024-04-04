# Analysis

This folder contain plotting code for Figures in the paper. Some scripts require Napari to run: `pip install "napari[all]"`.

## Figure 1
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


## Figure 5
* Figure 5a: [scripts/09_analyze.sh](../scripts/09_analyze.sh)

## Supplementary Figure 4
* See [01_eda.ipynb](./01_eda.ipynb)