#!/usr/bin/env bash

DATADIR="$HOME/DATA/INCEPTION_2020-CHD/Mice/DATA/CTs"

mousechd split_data \
    -metafile "$DATADIR/resampled/Imagine/metadata_20210203.csv" \
    -outdir "$DATADIR/labels" \
    -num_fold 5

mousechd split_data \
    -metafile "$DATADIR/resampled/Imagine/metadata_20210203.csv" \
    -outdir "$DATADIR/labels" \
    -val_size 0.2