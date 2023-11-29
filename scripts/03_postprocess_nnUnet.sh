DATADIR="$HOME/DATA/INCEPTION_2020-CHD/Mice/OUTPUTS/HeartSeg"

mousechd postprocess_nnUNet \
    -indir "$DATADIR/Imagine" \
    -outdir "$DATADIR/POSTPROCESSED"
