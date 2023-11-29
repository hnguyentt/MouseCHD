#!/usr/bin/env bash
module add cudnn/v8.7.0.84/cuda-11.8.0_520.61.05
export XLA_FLAGS=--xla_gpu_cuda_data_dir=/c7/shared/cuda/11.8.0_520.61.05
conda activate mousechd
DATADIR="$HOME/DATA/INCEPTION_2020-CHD/Mice/DATA"


#############
# 1ST BATCH #
#############
mousechd preprocess \
    -database "$DATADIR" \
    -imdir "CTs/raw/Imagine/images_20200206" \
    -maskdir "CTs/raw/Imagine/masks_20210115" \
    -masktype "TIF3d" \
    -metafile "$DATADIR/CTs/raw/Imagine/metadata/metadata_20210203.csv" \
    -sep ";" \
    -outdir "$DATADIR/CTs/processed1/Imagine"

mousechd prepare_nnUNet_data \
    -datadir "$DATADIR/CTs/processed/Imagine" \
    -seed 42 \
    -nnunet_dir "$HOME/DATA/nnUnet/nnUNet_raw/nnUNet_raw_data/Task103_MouseHeartImagine" \
    -test_size 0.2 \
    -save_test True

#############
# 2ND BATCH #
#############
mousechd preprocess \
    -database "$DATADIR" \
    -imdir "CTs/raw/Imagine/images_20200206" \
    -maskdir "$DATADIR/CTs/raw/Imagine/masks_20210708" \
    -masktype TIF2d \
    -outdir "$DATADIR/CTs/processed1/Imagine"

mousechd prepare_nnUNet_data \
    -datadir "$DATADIR/CTs/processed/Imagine" \
    -seed 42 \
    -nnunet_dir "$HOME/DATA/nnUnet/nnUNet_raw/nnUNet_raw_data/Task113_MouseHeartImagine20210807" \
    -continue_from "Task103_MouseHeartImagine" \
    -test_size 0.4

############
# NEW DATA #
############

# follow up
srun -J "followup" -c 4 mousechd preprocess \
    -database "$DATADIR" \
    -imdir "CTs/raw/Imagine/images_followup" \
    -metafile "$DATADIR/CTs/raw/Imagine/metadata/metadata_followup.csv" \
    -outdir "$DATADIR/CTs/processed/followup"

# Amaia
srun -J "Amaia" -c 4 mousechd preprocess \
    -database "$DATADIR" \
    -imdir "CTs/raw/Imagine/images_Amaia" \
    -metafile "$DATADIR/CTs/raw/Imagine/metadata/metadata_Amaia.csv" \
    -outdir "$DATADIR/CTs/processed/Amaia"

