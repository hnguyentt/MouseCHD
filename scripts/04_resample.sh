#!/usr/bin/env bash
####################
# RUN ON INCEPTION #
####################
module add cudnn/v8.7.0.84/cuda-11.8.0_520.61.05
conda activate mousechd
DATADIR="$HOME/DATA/INCEPTION_2020-CHD/Mice/DATA/CTs"
OUTDIR="$HOME/DATA/INCEPTION_2020-CHD/Mice/OUTPUTS/HeartSeg"

# Imagine
srun -J "Imagine" -c 4 mousechd resample \
    -imdir "$DATADIR/processed/Imagine/images" \
    -maskdir "$OUTDIR/Imagine" \
    -outdir "$DATADIR/resampled/Imagine" \
    -metafile "$DATADIR/processed/Imagine/metadata_20210203.csv" \
    -save_images 1

# followup
srun -J "followup" -c 4 mousechd resample \
    -imdir "$DATADIR/processed/followup/images" \
    -maskdir "$OUTDIR/followup" \
    -outdir "$DATADIR/resampled/followup" \
    -metafile "$DATADIR/processed/followup/metadata_followup.csv" \
    -save_images 1

mousechd create_label_df \
    -metafile "$DATADIR/resampled/followup/metadata_followup.csv" \
    -outdir "$DATADIR/labels" \
    -basename "followup"

# Amaia
srun -J "Amaia" -c 4 mousechd resample \
    -imdir "$DATADIR/processed/Amaia/images" \
    -maskdir "$OUTDIR/Amaia" \
    -outdir "$DATADIR/resampled/Amaia" \
    -metafile "$DATADIR/processed/Amaia/metadata_Amaia.csv" \
    -save_images 1

mousechd create_label_df \
    -metafile "$DATADIR/resampled/followup/metadata_followup.csv" \
    -outdir "$DATADIR/labels" \
    -basename "Amaia"

#================================//================================#
##################
# RUN ON MAESTRO #
##################
export PATH="/pasteur/appa/homes/hnguyent/miniconda3/bin:$PATH"
module load cuda/11.8
module load cudnn/11.x-v8.7.0.84
source activate mousechd
DATADIR="$HOME/DATA/INCEPTION_2020-CHD/Mice/DATA/CTs"
OUTDIR="$HOME/DATA/INCEPTION_2020-CHD/Mice/OUTPUTS/HeartSeg"



