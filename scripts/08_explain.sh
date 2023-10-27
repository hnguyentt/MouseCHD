module add cudnn/v8.7.0.84/cuda-11.8.0_520.61.05
export XLA_FLAGS=--xla_gpu_cuda_data_dir=/c7/shared/cuda/11.8.0_520.61.05
conda activate mousechd
WORKDIR="$HOME/DATA/INCEPTION_2020-CHD/Mice"


####################
# Generate GradCAM #
####################
F="F1"
srun -J "Imagine-$F" --pty --gres=gpu:1 -c 4 mousechd explain \
    -exp_dir "$WORKDIR/OUTPUTS/Classifier/$F" \
    -imdir "$WORKDIR/DATA/CTs/resampled/Imagine/images" \
    -outdir "$WORKDIR/OUTPUTS/GradCAMs/Imagine" \
    -label_path "$WORKDIR/DATA/CTs/labels/base/5folds/$F/test.csv"


srun -J "followup" --pty --gres=gpu:1 -c 4 mousechd explain \
    -exp_dir "$WORKDIR/OUTPUTS/Classifier/all-in" \
    -imdir "$WORKDIR/DATA/CTs/resampled/followup/images" \
    -outdir "$WORKDIR/OUTPUTS/GradCAMs/followup"


srun -J "Amaia" --pty --gres=gpu:1 -c 4 mousechd explain \
    -exp_dir "$WORKDIR/OUTPUTS/Classifier/retrain" \
    -imdir "$WORKDIR/DATA/CTs/resampled/Amaia/images" \
    -outdir "$WORKDIR/OUTPUTS/GradCAMs/Amaia"


