####################
# RUN ON INCEPTION #
####################
module add cudnn/v8.7.0.84/cuda-11.8.0_520.61.05
export XLA_FLAGS=--xla_gpu_cuda_data_dir=/c7/shared/cuda/11.8.0_520.61.05
conda activate mousechd
WORKDIR="$HOME/DATA/INCEPTION_2020-CHD/Mice"
SRCDIR="$HOME/DATA/INCEPTION_2020-CHD/Mice/projects/MouseCHD"

F="F2"
srun -J "$F" --pty --gres=gpu:1 -c 4 mousechd train_clf \
    -exp_dir "$WORKDIR/OUTPUTS" \
    -exp "Classifier/$F" \
    -data_dir "$WORKDIR/DATA/CTs/resampled/Imagine/images_x5" \
    -label_dir "$WORKDIR/DATA/CTs/labels/x5/5folds/$F" \
    -test_path "$WORKDIR/DATA/CTs/labels/base/5folds/$F/test.csv" \
    -testdir "$WORKDIR/DATA/CTs/resampled/Imagine/images" \
    -configs "$SRCDIR/assets/configs/configs.json" \
    -log_dir "$WORKDIR/LOGS" -evaluate "all"

# All data
srun -J "all-in" --pty --gres=gpu:1 -c 4 mousechd train_clf \
    -exp_dir "$WORKDIR/OUTPUTS" \
    -exp "Classifier/all-in" \
    -data_dir "$WORKDIR/DATA/CTs/resampled/Imagine" \
    -label_dir "$WORKDIR/DATA/CTs/labels/x5_base/1fold" \
    -test_path "$WORKDIR/DATA/CTs/labels/followup.csv" \
    -testdir "$WORKDIR/DATA/CTs/resampled/followup/images" \
    -configs "$SRCDIR/assets/configs/configs.json" \
    -log_dir "$WORKDIR/LOGS" -evaluate "all"

srun -J "retrain" --pty --gres=gpu:1 -c 4 mousechd train_clf \
    -exp_dir "$WORKDIR/OUTPUTS" \
    -exp "Classifier/retrain" \
    -data_dir "$WORKDIR/DATA/CTs/resampled/followup" \
    -label_dir "$WORKDIR/DATA/CTs/labels/x5_base/retrain" \
    -test_path "$WORKDIR/DATA/CTs/labels/Amaia.csv" \
    -testdir "$WORKDIR/DATA/CTs/resampled/Amaia/images" \
    -configs "$SRCDIR/assets/configs/configs.json" \
    -log_dir "$WORKDIR/LOGS" -evaluate "all"


##################
# RUN ON MAESTRO #
##################
export PATH="/pasteur/appa/homes/hnguyent/miniconda3/bin:$PATH"
module load cuda/11.8
module load cudnn/11.x-v8.7.0.84
export XLA_FLAGS=--xla_gpu_cuda_data_dir=/opt/gensoft/exe/cuda/11.8
source activate mousechd
