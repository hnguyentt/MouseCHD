DATADIR="$HOME/DATA/INCEPTION_2020-CHD/Mice/OUTPUTS/refactor/CTs/HeartSeg/nnUnet/Task113_MouseHeartImagine20210807"

mousechd postprocess_nnUNet \
    -indir "$DATADIR/TIF/ensemble" \
    -outdir "$DATADIR/TIF/Final"
