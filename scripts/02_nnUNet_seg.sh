###############
# Test nnUNet #
###############
PROCESSDIR="$HOME/DATA/INCEPTION_2020-CHD/Mice/DATA/CTs/processed"
OUTDIR="$HOME/DATA/INCEPTION_2020-CHD/Mice/OUTPUTS/HeartSeg"

# Imagine
srun -J "test_nnunet" --pty --gres=gpu:2 mousechd segment \
-indir "$PROCESSDIR/Imagine/images" \
-outdir "$OUTDIR/Imagine"

# Amaia
srun -J "test_nnunet" --pty --gres=gpu:2 mousechd segment \
-indir "$PROCESSDIR/Amaia/images" \
-outdir "$OUTDIR/Amaia"

# followup
srun -J "test_nnunet" --pty --gres=gpu:2 mousechd segment \
-indir "$PROCESSDIR/followup/images" \
-outdir "$OUTDIR/followup"
