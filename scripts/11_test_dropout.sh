#!/bin/bash

WORKDIR="$HOME/DATA/INCEPTION_2020-CHD/Mice"
echo $1

if [ "$1" == "all-in" ]; then
    echo $2
    for i in $(seq -w 1 50); do
        # Run the singularity command with the current value of i
        echo $i
        singularity exec --nv mousechd.sif python /master/home/hnguyent/DATA/INCEPTION_2020-CHD/Mice/projects/MouseCHD/scripts/bootstrap.py \
            -model_dir "$WORKDIR/OUTPUTS/Classifier/all-in" \
            -ckpt "best_model.hdf5" \
            -imdir "$WORKDIR/DATA/CTs/resampled/$2/images" \
            -label "$WORKDIR/DATA/CTs/labels/$2.csv" \
            -savename "$2/bootstrap_$i.csv" \
            -stage "eval" \
            -batch_size 8
    done

else

    # Loop from 01 to 50
    for i in $(seq -w 1 50); do
        # Run the singularity command with the current value of i
        echo $i
        singularity exec --nv mousechd.sif python /master/home/hnguyent/DATA/INCEPTION_2020-CHD/Mice/projects/MouseCHD/scripts/test_dropoutp.py \
            -model_dir "$WORKDIR/OUTPUTS/Classifier" \
            -fold "$1" \
            -ckpt "best_model.hdf5" \
            -imdir "$WORKDIR/DATA/CTs/resampled/Imagine/images" \
            -label "$WORKDIR/DATA/CTs/labels/base/5folds/$1/test.csv" \
            -savename "bootstrap/bootstrap_$i.csv" \
            -stage "eval" \
            -batch_size 8
    done
fi
