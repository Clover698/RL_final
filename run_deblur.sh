#!/bin/bash +e

# Check if an input is provided
if [ -z "$1" ]; then
  echo "Usage: $0 <input_root>"
  exit 1
fi
# Assign the first argument to INPUT_ROOT
INPUT_ROOT="$1"

# Define common arguments
CONFIG="celeba_hq.yml"
DOC="celeba"
TIMESTEPS=20
ETA=0.85
ETAB=1
SIGMA_0=0.0
IMAGE_FOLDER="celeba_hq_deblur_sigma_0.0"

# Training and evaluation for "deblur_uni"
DEG="deblur_uni"
echo "Starting training and evaluation for ${DEG}..."
for TARGET_STEPS in 5 10 20; do
    python train.py --ni --config $CONFIG --doc $DOC --timesteps $TIMESTEPS \
        --eta $ETA --etaB $ETAB --deg $DEG --sigma_0 $SIGMA_0 \
        -i $IMAGE_FOLDER --target_steps $TARGET_STEPS --input_root $INPUT_ROOT

    python train.py --ni --config $CONFIG --doc $DOC --timesteps $TIMESTEPS \
        --eta $ETA --etaB $ETAB --deg $DEG --sigma_0 $SIGMA_0 \
        -i $IMAGE_FOLDER --second_stage --target_steps $TARGET_STEPS --input_root $INPUT_ROOT

    python eval.py --ni --config $CONFIG --doc $DOC --timesteps $TIMESTEPS \
        --eta $ETA --etaB $ETAB --deg $DEG --sigma_0 $SIGMA_0 \
        -i $IMAGE_FOLDER --target_steps $TARGET_STEPS --eval_model_name ${DEG}_2agent_A2C_${TARGET_STEPS} --input_root $INPUT_ROOT
done

echo "Finished training and evaluation for ${DEG}."