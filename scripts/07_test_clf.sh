####################
# RUN ON INCEPTION #
####################
module add cudnn/v8.7.0.84/cuda-11.8.0_520.61.05
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/c7/shared/cudnn/v8.7.0.84/cuda-11/lib
export XLA_FLAGS=--xla_gpu_cuda_data_dir=/c7/shared/cuda/11.8.0_520.61.05
conda activate mousechd
WORKDIR="$HOME/DATA/INCEPTION_2020-CHD/Mice"

#Amaia
F="F1"
srun -J "Amaia" --pty --gres=gpu:1 -c 4 mousechd test_clf \
    -model_dir "$WORKDIR/OUTPUTS/Classifier" \
    -fold "$F" \
    -ckpt "all" \
    -imdir "$WORKDIR/DATA/CTs/resampled/Amaia/images" \
    -label "$WORKDIR/DATA/CTs/labels/Amaia.csv" \
    -stage "eval" -batch_size 8

srun -J "Amaia" --pty --gres=gpu:1 -c 4 mousechd test_clf \
    -model_dir "$WORKDIR/OUTPUTS/Classifier/all-in" \
    -ckpt "all" \
    -imdir "$WORKDIR/DATA/CTs/resampled/Amaia/images" \
    -label "$WORKDIR/DATA/CTs/labels/Amaia.csv" \
    -stage "eval" -batch_size 8

srun -J "Amaia" --pty --gres=gpu:1 -c 4 mousechd test_clf \
    -model_dir "$WORKDIR/OUTPUTS/Classifier/retrain" \
    -ckpt "all" \
    -imdir "$WORKDIR/DATA/CTs/resampled/Amaia/images" \
    -label "$WORKDIR/DATA/CTs/labels/Amaia.csv" \
    -stage "eval" -batch_size 8

# followup
F="F1"
srun -J "$F"_followup --pty --gres=gpu:1 -c 4 mousechd test_clf \
    -model_dir "$WORKDIR/OUTPUTS/Classifier" \
    -fold "$F" \
    -ckpt "all" \
    -imdir "$WORKDIR/DATA/CTs/resampled/followup/images" \
    -label "$WORKDIR/DATA/CTs/labels/followup.csv" \
    -stage "eval" -batch_size 8

srun -J "followup" --pty --gres=gpu:1 -c 4 mousechd test_clf \
    -model_dir "$WORKDIR/OUTPUTS/Classifier/all-in" \
    -ckpt "all" \
    -imdir "$WORKDIR/DATA/CTs/resampled/followup/images" \
    -label "$WORKDIR/DATA/CTs/labels/followup.csv" \
    -stage "eval" -batch_size 8

# Imagine
F="F1"
srun -J "$F"_Imagine --pty --gres=gpu:1 -c 4 mousechd test_clf \
    -model_dir "$WORKDIR/OUTPUTS/Classifier" \
    -fold "$F" \
    -ckpt "all" \
    -imdir "$WORKDIR/DATA/CTs/resampled/Imagine/images" \
    -label "$WORKDIR/DATA/CTs/labels/base/5folds/$F/test.csv" \
    -stage "eval" -batch_size 8

srun -J "$F"_Imagine --pty --gres=gpu:1 -c 4 mousechd test_clf \
    -model_dir "$WORKDIR/OUTPUTS/Classifier/all-in" \
    -ckpt "all" \
    -imdir "$WORKDIR/DATA/CTs/resampled/Imagine/images" \
    -label "$WORKDIR/DATA/CTs/labels/base/5folds/$F/test.csv" \
    -stage "eval" -batch_size 8

#==============================//==============================#

##################
# RUN ON MAESTRO #
##################
export PATH="/pasteur/appa/homes/hnguyent/miniconda3/bin:$PATH"
module load cuda/11.8
module load cudnn/11.x-v8.7.0.84
source activate mousechd
WORKDIR="$HOME/DATA/zeus/hnguyent/DATA/MouseCHD"







