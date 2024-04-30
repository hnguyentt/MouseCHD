#############
# CONSTANTS #
#############
RAWDIR="$HOME/DATA/INCEPTION_2020-CHD/Mice/DATA/CTs/raw/Imagine"
PROCESSDIR="$HOME/DATA/INCEPTION_2020-CHD/Mice/DATA/CTs/processed"
RESAMPLEDIR="$HOME/DATA/INCEPTION_2020-CHD/Mice/DATA/CTs/resampled"
OUTDIR="$HOME/DATA/INCEPTION_2020-CHD/Mice/OUTPUTS"
SAVEDIR="$HOME/DATA/INCEPTION_2020-CHD/Mice/PAPER/FIGURES"

# For visualization: pip install "napari[all]"

########################
# Figure 01 components #
########################
# <----- 20220926 ----->
# 3D
## Stage P0 w/o bbx
mousechd viz3d_stages \
    -imdir "$PROCESSDIR/Imagine/images" \
    -maskdir "$PROCESSDIR/Imagine/heart-masks" \
    -stage "P0" \
    -imname "NH_221c" \
    -savedir "$SAVEDIR/Fig01" \
    -zoom 2.8 \
    -cam_angles "(0,30,15)" \
    -crop 1 \
    -left_off 0.17 \
    -right_off 0.17

## Stage E18.5 w/o bbx
mousechd viz3d_stages \
    -imdir "$PROCESSDIR/Imagine/images" \
    -maskdir "$PROCESSDIR/Imagine/heart-masks" \
    -stage "E18.5" \
    -imname "NH_1045" \
    -savedir "$SAVEDIR/Fig01" \
    -zoom 2.8 \
    -cam_angles "(0,30,15)" \
    -crop 1 \
    -left_off 0.17 \
    -right_off 0.17

## Stage P0 w/ bbx
mousechd viz3d_stages \
    -imdir "$PROCESSDIR/Imagine/images" \
    -maskdir "$PROCESSDIR/Imagine/heart-masks" \
    -stage "P0" \
    -imname "NH_221c" \
    -savedir "$SAVEDIR/Fig01" \
    -suffix "bbx" \
    -bbx "both" \
    -zoom 2.8 \
    -cam_angles "(0,30,15)" \
    -crop 1 \
    -left_off 0.17 \
    -right_off 0.17

## Stage E18.5 w/ bbx
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
    -crop 1 \
    -left_off 0.17 \
    -right_off 0.17

## Stage P0 with axial bbx
mousechd viz3d_stages \
    -imdir "$PROCESSDIR/Imagine/images" \
    -maskdir "$PROCESSDIR/Imagine/heart-masks" \
    -stage "P0" \
    -imname "NH_221c" \
    -savedir "$SAVEDIR/Fig01" \
    -suffix "bbxaxial" \
    -bbx "axial" \
    -zoom 2.8 \
    -cam_angles "(0,30,15)" \
    -crop 1 \
    -left_off 0.17 \
    -right_off 0.17

## Stage E18.5 with axial bbx
mousechd viz3d_stages \
    -imdir "$PROCESSDIR/Imagine/images" \
    -maskdir "$PROCESSDIR/Imagine/heart-masks" \
    -stage "E18.5" \
    -imname "NH_1045" \
    -savedir "$SAVEDIR/Fig01" \
    -suffix "bbxaxial" \
    -bbx "axial" \
    -zoom 2.8 \
    -cam_angles "(0,30,15)" \
    -crop 1 \
    -left_off 0.17 \
    -right_off 0.17

## Stage P0 with heart bbx
mousechd viz3d_stages \
    -imdir "$PROCESSDIR/Imagine/images" \
    -maskdir "$PROCESSDIR/Imagine/heart-masks" \
    -stage "P0" \
    -imname "NH_221c" \
    -savedir "$SAVEDIR/Fig01" \
    -suffix "bbxheart" \
    -bbx "heart" \
    -zoom 2.8 \
    -cam_angles "(0,30,15)" \
    -crop 1 \
    -left_off 0.17 \
    -right_off 0.17

## Stage E18.5 with heart bbx
mousechd viz3d_stages \
    -imdir "$PROCESSDIR/Imagine/images" \
    -maskdir "$PROCESSDIR/Imagine/heart-masks" \
    -stage "E18.5" \
    -imname "NH_1045" \
    -savedir "$SAVEDIR/Fig01" \
    -suffix "bbxheart" \
    -bbx "heart" \
    -zoom 2.8 \
    -cam_angles "(0,30,15)" \
    -crop 1 \
    -left_off 0.17 \
    -right_off 0.17

## Stage E17.5 with heart bbx
mousechd viz3d_stages \
    -imdir "$PROCESSDIR/Amaia/images" \
    -maskdir "$HOME/DATA/INCEPTION_2020-CHD/Mice/OUTPUTS/HeartSeg/Amaia" \
    -stage "E17.5" \
    -imname "C_168" \
    -savedir "$SAVEDIR/Fig05/Components" \
    -suffix "bbx" \
    -bbx "both" \
    -zoom 2.8 \
    -cam_angles "(0,30,15)" \
    -crop 1 \
    -left_off 0.17 \
    -right_off 0.17

# 2D
## Stage P0 axial with heart annotation
mousechd viz_stacks \
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

# 07/09/2023
mousechd viz_stacks \
    -impath "$PROCESSDIR/Imagine/images/NH_221c" \
    -crop "axial" \
    -annotate 1 \
    -heart_cnt 1 \
    -maskpath "$PROCESSDIR/Imagine/heart-masks/NH_221c" \
    -pad "(0,1,1)" \
    -num 5 \
    -trans_val 30 \
    -savedir "$SAVEDIR/Fig01" \
    -plot_zline 1 \
    -zline_color 'b' \
    -plot_border 1 \
    -color 'b'

## Stage P0 axial w/o heart annotation
mousechd viz_stacks \
    -impath "$PROCESSDIR/Imagine/images/NH_221c" \
    -maskpath "$PROCESSDIR/Imagine/heart-masks/NH_221c" \
    -crop "axial" \
    -pad "(1,1,1)" \
    -num 5 \
    -trans_val 30 \
    -savedir "$SAVEDIR/Fig01" \
    -plot_zline 1 \
    -zline_color 'b' \
    -plot_border 1 \
    -color 'b'

## Stage P0 heart
mousechd viz_stacks \
    -impath "$PROCESSDIR/Imagine/images/NH_221c" \
    -maskpath "$PROCESSDIR/Imagine/heart-masks/NH_221c" \
    -crop "heart" \
    -pad "(1,1,1)" \
    -num 5 \
    -trans_val 10 \
    -savedir "$SAVEDIR/Fig01" \
    -linewidth 6 \
    -plot_zline 1 \
    -zline_color 'r' \
    -plot_border 1 \
    -color 'r'

### 07/09/2023
mousechd viz_stacks \
    -impath "$PROCESSDIR/Imagine/images/NH_221c" \
    -maskpath "$PROCESSDIR/Imagine/heart-masks/NH_221c" \
    -crop "heart" \
    -heart_cnt 1 \
    -pad "(0,1,1)" \
    -num 5 \
    -trans_val 10 \
    -savedir "$SAVEDIR/Fig01" \
    -linewidth 6 \
    -plot_zline 1 \
    -zline_color 'r' \
    -plot_border 1 \
    -color 'r'

## Stage E18.5 axial with heart annotation
mousechd viz_stacks \
    -impath "$PROCESSDIR/Imagine/images/NH_1045" \
    -crop "axial" \
    -annotate 1 \
    -maskpath "$PROCESSDIR/Imagine/heart-masks/NH_1045" \
    -pad "(1,1,1)" \
    -num 5 \
    -trans_val 30 \
    -savedir "$SAVEDIR/Fig01" \
    -plot_zline 1 \
    -zline_color 'b' \
    -plot_border 1 \
    -color 'b'
### 07/09/2023
mousechd viz_stacks \
    -impath "$PROCESSDIR/Imagine/images/NH_1045" \
    -crop "axial" \
    -annotate 1 \
    -heart_cnt 1 \
    -maskpath "$PROCESSDIR/Imagine/heart-masks/NH_1045" \
    -pad "(0,1,1)" \
    -num 5 \
    -trans_val 30 \
    -savedir "$SAVEDIR/Fig01" \
    -plot_zline 1 \
    -zline_color 'b' \
    -plot_border 1 \
    -color 'b'

## Stage E18.5 axial w/o heart annotation
mousechd viz_stacks \
    -impath "$PROCESSDIR/Imagine/images/NH_1045" \
    -maskpath "$PROCESSDIR/Imagine/heart-masks/NH_1045" \
    -crop "axial" \
    -pad "(1,1,1)" \
    -num 5 \
    -trans_val 30 \
    -savedir "$SAVEDIR/Fig01" \
    -plot_zline 1 \
    -zline_color 'b' \
    -plot_border 1 \
    -color 'b'

## Stage E18.5 heart
mousechd viz_stacks \
    -impath "$PROCESSDIR/Imagine/images/NH_1045" \
    -maskpath "$PROCESSDIR/Imagine/heart-masks/NH_1045" \
    -crop "heart" \
    -pad "(1,1,1)" \
    -num 5 \
    -trans_val 10 \
    -savedir "$SAVEDIR/Fig01" \
    -linewidth 6 \
    -plot_zline 1 \
    -zline_color 'r' \
    -plot_border 1 \
    -color 'r'
### 07/09/2023
mousechd viz_stacks \
    -impath "$PROCESSDIR/Imagine/images/NH_1045" \
    -maskpath "$PROCESSDIR/Imagine/heart-masks/NH_1045" \
    -crop "heart" \
    -heart_cnt 1 \
    -pad "(0,1,1)" \
    -num 5 \
    -trans_val 10 \
    -savedir "$SAVEDIR/Fig01" \
    -linewidth 6 \
    -plot_zline 1 \
    -zline_color 'r' \
    -plot_border 1 \
    -color 'r'
### E17.5
mousechd viz_stacks \
    -impath "$PROCESSDIR/Amaia/images/C_100" \
    -crop "axial" \
    -annotate 1 \
    -heart_cnt 1 \
    -maskpath "$HOME/DATA/INCEPTION_2020-CHD/Mice/OUTPUTS/HeartSeg/Amaia/C_100" \
    -pad "(0,1,1)" \
    -num 5 \
    -trans_val 30 \
    -savedir "$SAVEDIR/Fig05/Components" \
    -plot_zline 1 \
    -zline_color 'b' \
    -plot_border 1 \
    -color 'b'


mousechd viz_stacks \
    -impath "$PROCESSDIR/Amaia/images/C_100" \
    -maskpath "$HOME/DATA/INCEPTION_2020-CHD/Mice/OUTPUTS/HeartSeg/Amaia/C_100" \
    -crop "heart" \
    -heart_cnt 1 \
    -pad "(0,2,2)" \
    -num 5 \
    -trans_val 10 \
    -savedir "$SAVEDIR/Fig05/Components" \
    -linewidth 6 \
    -plot_zline 1 \
    -zline_color 'r' \
    -plot_border 1 \
    -color 'r'

# Venn diagram and contingent matrix
mousechd viz_eda \
    -term_path "$RAWDIR/metadata/terminology_20201217.csv" \
    -meta_path "$PROCESSDIR/Imagine/metadata_20210203.csv" \
    -savedir "$SAVEDIR/Fig01"

########################
# Figure 02 components #
########################
# 2D
## full heart
mousechd viz_stacks \
    -impath "$RESAMPLEDIR/Imagine/images/NH_1045" \
    -crop "none" \
    -annotate 0 \
    -maskpath "$RESAMPLEDIR/Imagine/images/NH_1045" \
    -pad "(5,5,5)" \
    -num 7 \
    -trans_val 10 \
    -savedir "$SAVEDIR/Fig02" \
    -linewidth 8 \
    -plot_zline 1 \
    -zline_color 'k' \
    -plot_border 1 \
    -color 'orange;g;indigo;y;m;orange;g'

## resample axial5
i=1
color='orange'
i=2
color='g'
i=3
color='indigo'
i=4
color='y'
i=5
color='m'
mousechd viz_stacks \
    -impath "$RESAMPLEDIR/Imagine/images_x5/NH_1045_0$i" \
    -crop "none" \
    -annotate 0 \
    -maskpath "$RESAMPLEDIR/Imagine/images_x5/NH_1045_0$i" \
    -pad "(1,1,1)" \
    -num 5 \
    -trans_val 10 \
    -savedir "$SAVEDIR/Fig02" \
    -linewidth 5 \
    -plot_zline 1 \
    -zline_color "$color" \
    -plot_border 1 \
    -color "$color"

# 3D
## Full thorax (twilight colormap)
mousechd viz3d_stages \
    -imdir "$PROCESSDIR/Imagine/images" \
    -maskdir "$PROCESSDIR/Imagine/heart-masks" \
    -stage "E18.5" \
    -imname "NH_1045" \
    -savedir "$SAVEDIR/Fig02" \
    -zoom 2.8 \
    -cam_angles "(0,30,15)" \
    -heart_cmap "twilight" \
    -heart_opac 0.25 \
    -crop 1 \
    -left_off 0.24 \
    -right_off 0.24

## Full thorax (turbo colormap)
mousechd viz3d_stages \
    -imdir "$PROCESSDIR/Imagine/images" \
    -maskdir "$PROCESSDIR/Imagine/heart-masks" \
    -stage "E18.5" \
    -imname "NH_1045" \
    -suffix "turbo" \
    -savedir "$SAVEDIR/Fig02" \
    -zoom 2.8 \
    -cam_angles "(0,30,15)" \
    -heart_cmap "turbo" \
    -heart_opac 0.25 \
    -crop 1 \
    -left_off 0.24 \
    -right_off 0.24

# heart w/o bbx
mousechd viz3d_stages \
    -imdir "$PROCESSDIR/Imagine/images" \
    -maskdir "$PROCESSDIR/Imagine/heart-masks" \
    -stage "E18.5" \
    -imname "NH_1045" \
    -suffix "heart" \
    -savedir "$SAVEDIR/Fig02" \
    -zoom 2.8 \
    -cam_angles "(0,30,15)" \
    -im_opac 0 \
    -heart_cmap "red" \
    -heart_opac 1 \
    -crop 1 \
    -left_off 0.24 \
    -right_off 0.24

# heart with bbx
mousechd viz3d_stages \
    -imdir "$PROCESSDIR/Imagine/images" \
    -maskdir "$PROCESSDIR/Imagine/heart-masks" \
    -bbx "heart" \
    -pad "(1,5,5)" \
    -stage "E18.5" \
    -imname "NH_1045" \
    -suffix "bbxheart" \
    -savedir "$SAVEDIR/Fig02" \
    -zoom 2.8 \
    -cam_angles "(0,30,15)" \
    -im_opac 0 \
    -heart_cmap "red" \
    -heart_opac 1 \
    -crop 1 \
    -left_off 0.24 \
    -right_off 0.24

mousechd viz3d_stages \
    -imdir "$PROCESSDIR/Imagine/images" \
    -maskdir "$PROCESSDIR/Imagine/heart-masks" \
    -stage "E18.5" \
    -imname "NH_1045" \
    -suffix "heart" \
    -savedir "$SAVEDIR/Fig02" \
    -zoom 2.8 \
    -cam_angles "(0,30,15)" \
    -im_opac 0 \
    -heart_cmap "turbo" \
    -heart_opac 1 \
    -crop 1 \
    -left_off 0.24 \
    -right_off 0.24


####################
# Sup01 Components #
####################
# IMNAME='NH_314m'
# IMNAME='NH_1020'
# IMNAME='NH_233C'
IMNAME="NH_256c"
IMNAME="NH_224C"
IMNAME="NH_294m"
mousechd viz3d_seg \
    -imdir "$PROCESSDIR/Imagine/images" \
    -maskdir "$PROCESSDIR/Imagine/heart-masks" \
    -preddir "$OUTDIR/HeartSeg/Imagine" \
    -imname "$IMNAME" \
    -savedir "$SAVEDIR/Sup01/Components"

########################################
# Visualize different resample methods #
########################################
mousechd viz3d_views \
    -imdir "$HOME/DATA/INCEPTION_2020-CHD/Mice/DATA/CTs/arxiv/resampled/Imagine_M28_1/masked_images" \
    -imname "N_256c_01.nii.gz" \
    -outdir "$SAVEDIR/Extra"


#########
# Sup04 #
#########
mousechd viz_grad \
    -imdir "$OUTDIR/GradCAMs/Imagine/positives" \
    -savedir "$SAVEDIR/Sup04" \
    -hearts "NH_333m;NH_255m;NH_282"
