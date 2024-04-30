RAWDIR="$HOME/DATA/INCEPTION_2020-CHD/Mice/DATA/CTs/raw/Imagine"
PROCESSDIR="$HOME/DATA/INCEPTION_2020-CHD/Mice/DATA/CTs/processed"
RESAMPLEDIR="$HOME/DATA/INCEPTION_2020-CHD/Mice/DATA/CTs/resampled"
OUTDIR="$HOME/DATA/INCEPTION_2020-CHD/Mice/OUTPUTS"
SAVEDIR="$HOME/DATA/INCEPTION_2020-CHD/Mice/PAPER/FIGURES"
#################################
# Round 1: Add scale for images #
#################################

#**********#
# Figure 1 #
#**********#
# -----------------------------------------------#
# Stage E18.5 3D
mousechd viz3d_stages \
    -imdir "$PROCESSDIR/Imagine/images" \
    -maskdir "$PROCESSDIR/Imagine/heart-masks" \
    -stage "E18.5" \
    -imname "NH_1045" \
    -savedir "$SAVEDIR/revision0/components" \
    -suffix "bbx" \
    -bbx "both" \
    -zoom 200 \
    -cam_angles "(0,30,15)"

# Stage E18.5 2D
mousechd viz_stacks \
    -impath "$PROCESSDIR/Imagine/images/NH_1045" \
    -crop "axial" \
    -annotate 1 \
    -heart_cnt 1 \
    -maskpath "$PROCESSDIR/Imagine/heart-masks/NH_1045" \
    -pad "(0,1,1)" \
    -num 5 \
    -trans_val 30 \
    -savedir "$SAVEDIR/revision0/components" \
    -plot_zline 1 \
    -zline_color 'b' \
    -plot_border 1 \
    -color 'b' \
    -scalebar '{"value": 4}'

mousechd viz_stacks \
    -impath "$PROCESSDIR/Imagine/images/NH_1045" \
    -maskpath "$PROCESSDIR/Imagine/heart-masks/NH_1045" \
    -crop "heart" \
    -heart_cnt 1 \
    -pad "(0,1,1)" \
    -num 5 \
    -trans_val 10 \
    -savedir "$SAVEDIR/revision0/components" \
    -linewidth 6 \
    -plot_zline 1 \
    -zline_color 'r' \
    -plot_border 1 \
    -color 'r' \
    -scalebar '{"value": 1.5, "linewidth": 7, "fontsize": 45, "h_pos": 2, "markersize": 18}'

# -----------------------------------------------#
# Stage P0
mousechd viz3d_stages \
    -imdir "$PROCESSDIR/Imagine/images" \
    -maskdir "$PROCESSDIR/Imagine/heart-masks" \
    -stage "P0" \
    -imname "NH_221c" \
    -savedir "$SAVEDIR/revision0/components" \
    -suffix "bbx" \
    -bbx "both" \
    -zoom 200 \
    -cam_angles "(0,30,15)"

# Stage P0 2D
mousechd viz_stacks \
    -impath "$PROCESSDIR/Imagine/images/NH_221c" \
    -crop "axial" \
    -annotate 1 \
    -heart_cnt 1 \
    -maskpath "$PROCESSDIR/Imagine/heart-masks/NH_221c" \
    -pad "(0,1,1)" \
    -num 5 \
    -trans_val 30 \
    -savedir "$SAVEDIR/revision0/components" \
    -plot_zline 1 \
    -zline_color 'b' \
    -plot_border 1 \
    -color 'b' \
    -scalebar '{"value": 4}'

mousechd viz_stacks \
    -impath "$PROCESSDIR/Imagine/images/NH_221c" \
    -maskpath "$PROCESSDIR/Imagine/heart-masks/NH_221c" \
    -crop "heart" \
    -heart_cnt 1 \
    -pad "(0,1,1)" \
    -num 5 \
    -trans_val 10 \
    -savedir "$SAVEDIR/revision0/components" \
    -linewidth 6 \
    -plot_zline 1 \
    -zline_color 'r' \
    -plot_border 1 \
    -color 'r' \
    -scalebar '{"value": 1.5, "linewidth": 7, "fontsize": 45, "h_pos": 2, "markersize": 18}'

#**********#
# Figure 2 #
#**********#
mousechd viz3d_stages \
    -imdir "$PROCESSDIR/Imagine/images" \
    -maskdir "$PROCESSDIR/Imagine/heart-masks" \
    -stage "E18.5" \
    -imname "NH_1045" \
    -savedir "$SAVEDIR/revision0/components" \
    -zoom 200 \
    -cam_angles "(0,30,15)" \
    -heart_cmap "twilight"

mousechd viz3d_stages \
    -imdir "$PROCESSDIR/Imagine/images" \
    -maskdir "$PROCESSDIR/Imagine/heart-masks" \
    -bbx "heart" \
    -pad "(1,5,5)" \
    -stage "E18.5" \
    -imname "NH_1045" \
    -suffix "bbxheart" \
    -savedir "$SAVEDIR/revision0/components" \
    -zoom 200 \
    -cam_angles "(0,30,15)" \
    -im_opac 0 \
    -heart_cmap "red" \
    -heart_opac 1

mousechd viz_stacks \
    -impath "$RESAMPLEDIR/Imagine/images/NH_1045" \
    -crop "none" \
    -annotate 0 \
    -maskpath "$RESAMPLEDIR/Imagine/images/NH_1045" \
    -pad "(5,5,5)" \
    -num 7 \
    -trans_val 10 \
    -savedir "$SAVEDIR/revision0/components" \
    -linewidth 8 \
    -plot_zline 1 \
    -zline_color 'k' \
    -plot_border 1 \
    -color 'orange;g;indigo;y;m;orange;g' \
    -scalebar '{"value": 1.5, "linewidth": 7, "fontsize": 45, "h_pos": 2, "markersize": 18}'

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
    -savedir "$SAVEDIR/revision0/components" \
    -linewidth 5 \
    -plot_zline 1 \
    -zline_color "$color" \
    -plot_border 1 \
    -color "$color" \
    -scalebar '{"value": 1.5, "linewidth": 7, "fontsize": 45, "h_pos": 2, "markersize": 18}'

# -----------------------------------------------#
# Stage E17.5
mousechd viz3d_stages \
    -imdir "$PROCESSDIR/Amaia/images" \
    -maskdir "$HOME/DATA/INCEPTION_2020-CHD/Mice/OUTPUTS/HeartSeg/Amaia" \
    -stage "E17.5" \
    -imname "C_100" \
    -savedir "$SAVEDIR/revision0/components" \
    -suffix "bbx" \
    -bbx "both" \
    -zoom 200 \
    -cam_angles "(0,30,15)"

mousechd viz_stacks \
    -impath "$PROCESSDIR/Amaia/images/C_100" \
    -crop "axial" \
    -annotate 1 \
    -heart_cnt 1 \
    -maskpath "$HOME/DATA/INCEPTION_2020-CHD/Mice/OUTPUTS/HeartSeg/Amaia/C_100" \
    -pad "(0,1,1)" \
    -num 5 \
    -trans_val 30 \
    -savedir "$SAVEDIR/revision0/components" \
    -plot_zline 1 \
    -zline_color 'b' \
    -plot_border 1 \
    -color 'b' \
    -scalebar '{"value": 4}'

mousechd viz_stacks \
    -impath "$PROCESSDIR/Amaia/images/C_100" \
    -maskpath "$HOME/DATA/INCEPTION_2020-CHD/Mice/OUTPUTS/HeartSeg/Amaia/C_100" \
    -crop "heart" \
    -heart_cnt 1 \
    -pad "(0,2,2)" \
    -num 5 \
    -trans_val 10 \
    -savedir "$SAVEDIR/revision0/components" \
    -linewidth 6 \
    -plot_zline 1 \
    -zline_color 'r' \
    -plot_border 1 \
    -color 'r' \
    -scalebar '{"value": 1.5, "linewidth": 7, "fontsize": 45, "h_pos": 2, "markersize": 18}'


#**********#
# Sup Fig1 #
#**********#
IMNAME="NH_256c"
IMNAME="NH_224C"
IMNAME="NH_294m"
mousechd viz3d_seg \
    -imdir "$PROCESSDIR/Imagine/images" \
    -maskdir "$PROCESSDIR/Imagine/heart-masks" \
    -preddir "$OUTDIR/HeartSeg/Imagine" \
    -imname "$IMNAME" \
    -savedir "$SAVEDIR/revision0/components"


#**********#
# Sup Fig5 #
#**********#
mousechd viz_grad \
    -imdir "$OUTDIR/GradCAMs/Imagine/positives/TP" \
    -savedir "$SAVEDIR/revision0/components" \
    -hearts "NH_333m;NH_255m;NH_282"