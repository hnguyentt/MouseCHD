####################
# RUN ON INCEPTION #
####################
module add cudnn/v8.7.0.84/cuda-11.8.0_520.61.05
export XLA_FLAGS=--xla_gpu_cuda_data_dir=/c7/shared/cuda/11.8.0_520.61.05
conda activate mousechd
WORKDIR="$HOME/DATA/INCEPTION_2020-CHD/Mice"
SRCDIR="$HOME/DATA/INCEPTION_2020-CHD/Mice/projects/MouseCHD"

F="F1"
srun -J "$F" --pty --gres=gpu:1 -c 4 mousechd train_clf \
    -exp_dir "$WORKDIR/OUTPUTS" \
    -exp "Classifier4/$F" \
    -data_dir "$WORKDIR/DATA/CTs/resampled/Imagine" \
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

# Revision
srun -J "retrain" --pty --gres=gpu:1 -c 4 mousechd train_clf \
    -exp_dir "$WORKDIR/OUTPUTS" \
    -exp "Classifier/retrain_refined" \
    -data_dir "$WORKDIR/DATA/CTs/resampled" \
    -label_dir "$WORKDIR/DATA/CTs/labels/x5_base/retrain_refined" \
    -test_path "$WORKDIR/DATA/CTs/labels/divergence.csv" \
    -testdir "$WORKDIR/DATA/CTs/resampled" \
    -configs "$SRCDIR/assets/configs/configs.json" \
    -log_dir "$WORKDIR/LOGS" -evaluate "all"

# Revision 1
srun -J "div_scratch" --pty --gres=gpu:1 singularity exec --nv mousechd.sif mousechd train_clf \
    -exp_dir "$WORKDIR/OUTPUTS" \
    -exp "Classifier/div_scratch" \
    -data_dir "$WORKDIR/DATA/CTs/resampled" \
    -label_dir "$WORKDIR/DATA/CTs/labels/x5/retrain_refined" \
    -test_path "$WORKDIR/DATA/CTs/labels/divergence.csv" \
    -testdir "$WORKDIR/DATA/CTs/resampled" \
    -configs "$SRCDIR/assets/configs/configs.json" \
    -log_dir "$WORKDIR/LOGS" -evaluate "all"

# I2
srun -J "I2" --pty --gres=gpu:1 singularity exec --nv mousechd.sif mousechd train_clf \
    -exp_dir "$WORKDIR/OUTPUTS" \
    -exp "Classifier/I2" \
    -data_dir "$WORKDIR/DATA/CTs/resampled/Imagine/images_x5" \
    -label_dir "$WORKDIR/DATA/CTs/labels/x5/revisions/I2" \
    -test_path "$WORKDIR/DATA/CTs/labels/base/revisions/I2/test.csv" \
    -testdir "$WORKDIR/DATA/CTs/resampled/Imagine/images" \
    -configs "$SRCDIR/assets/configs/configs.json" \
    -log_dir "$WORKDIR/LOGS" -evaluate "all"

# I3
srun -J "I2" --pty --gres=gpu:1 singularity exec --nv mousechd.sif mousechd train_clf \
    -exp_dir "$WORKDIR/OUTPUTS" \
    -exp "Classifier/I3" \
    -data_dir "$WORKDIR/DATA/CTs/resampled" \
    -label_dir "$WORKDIR/DATA/CTs/labels/x5/revisions/I3" \
    -test_path "$WORKDIR/DATA/CTs/labels/base/revisions/I3/test.csv" \
    -testdir "$WORKDIR/DATA/CTs/resampled/Imagine/images" \
    -configs "$SRCDIR/assets/configs/configs.json" \
    -log_dir "$WORKDIR/LOGS" -evaluate "all"

# WholeScanNew
F="F1"
singularity exec --nv mousechd.sif mousechd train_clf \
    -exp_dir "$WORKDIR/OUTPUTS" \
    -exp "WholeScanNew1/$F" \
    -data_dir "$WORKDIR/DATA/CTs/resampled/whole_scans/images_x5" \
    -label_dir "$WORKDIR/DATA/CTs/labels/x5/5folds/$F" \
    -configs "$SRCDIR/configs/configs.json" \
    -log_dir "$WORKDIR/LOGS"

singularity exec --nv mousechd.sif mousechd test_clf \
    -model_dir "$WORKDIR/OUTPUTS/WholeScanNew1/$F" \
    -ckpt "best_model.hdf5" \
    -imdir "$WORKDIR/DATA/CTs/resampled/whole_scans/images" \
    -label "$WORKDIR/DATA/CTs/labels/base/5folds/$F/test.csv" \
    -stage "eval" -batch_size 2

## On maestro
WORKDIR="/pasteur/appa/homes/hnguyent/DATA/zeus/hnguyent/DATA/MouseCHD"
SRCDIR="/pasteur/appa/homes/hnguyent/DATA/zeus/hnguyent/DATA/MouseCHD"
F=F3
srun -u -J "$F" -A gpulab -p gpulab --qos=gpu --gres=gpu:1 apptainer exec -B /pasteur --nv mousechd.sif mousechd train_clf \
    -exp_dir "$WORKDIR/OUTPUTS" \
    -exp "WholeScanNew/$F" \
    -data_dir "$WORKDIR/DATA/resampled/whole_scans/images_x5" \
    -label_dir "$WORKDIR/DATA/labels/x5/5folds/$F" \
    -test_path "$WORKDIR/DATA/labels/base/5folds/$F/test.csv" \
    -testdir "$WORKDIR/DATA/resampled/whole_scans/images" \
    -configs "$SRCDIR/configs.json" \
    -log_dir "$WORKDIR/LOGS" -evaluate "all"

srun -u -J "$F" -p gpu --qos=gpu --gres=gpu:1,gmem:48G apptainer exec -B /pasteur --nv mousechd.sif mousechd train_clf \
    -exp_dir "$WORKDIR/OUTPUTS" \
    -exp "WholeScanNew/$F" \
    -data_dir "$WORKDIR/DATA/resampled/whole_scans/images_x5" \
    -label_dir "$WORKDIR/DATA/labels/x5/5folds/$F" \
    -configs "$SRCDIR/configs.json" \
    -log_dir "$WORKDIR/LOGS"

srun -u -J "$F" -A gpulab -p gpulab --qos=gpu --gres=gpu:1 apptainer exec -B /pasteur --nv mousechd.sif mousechd test_clf \
    -model_dir "$WORKDIR/OUTPUTS/WholeScanNew1/$F" \
    -ckpt "best_model.hdf5" \
    -imdir "$WORKDIR/DATA/resampled/whole_scans/images" \
    -label "$WORKDIR/DATA/labels/base/5folds/$F/test.csv" \
    -stage "eval" -batch_size 2

##################
# RUN ON MAESTRO #
##################
export PATH="/pasteur/appa/homes/hnguyent/miniconda3/bin:$PATH"
module load cuda/11.8
module load cudnn/11.x-v8.7.0.84
export XLA_FLAGS=--xla_gpu_cuda_data_dir=/opt/gensoft/exe/cuda/11.8
source activate mousechd
