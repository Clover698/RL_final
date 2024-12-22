#!/bin/bash +e

# Define common arguments
CONFIG="celeba_hq.yml"
DOC="celeba"
ETA=0.85
SIGMA_0=0.
INPUT="celeba_hq_sr4_sigma_0.05"
INPUT_ROOT="/disk_195a/qiannnhui"

# Training and evaluation for "sr4"
for DEG in sr_bicubic deblur_uni
do
    echo "Starting training and evaluation for ${DEG}..."
    for TARGET_STEPS in 5 10 20; do
        python train.py --ni --config $CONFIG --path_y $DOC \
            --eta $ETA --deg $DEG --sigma_y $SIGMA_0 --deg_scale 4 \
            -i $INPUT --target_steps $TARGET_STEPS --input_root $INPUT_ROOT

        python train.py --ni --config $CONFIG --path_y $DOC \
            --eta $ETA --deg $DEG --sigma_y $SIGMA_0 --deg_scale 4 \
            -i $INPUT --second_stage --target_steps $TARGET_STEPS --input_root $INPUT_ROOT

        python eval.py --ni --config $CONFIG --path_y $DOC \
            --eta $ETA --deg $DEG --sigma_y $SIGMA_0 --deg_scale 4 \
            -i $INPUT --target_steps $TARGET_STEPS --eval_model_name ${DEG}_2agent_A2C_${TARGET_STEPS} --input_root $INPUT_ROOT
    done

    echo "Finished training and evaluation for ${DEG}."
done

# Please modify the dataset path of imagenet in datasets/__init__.py (line 176 & 180). You can download the dataset from http://image-net.org/download-images. Note the dataset is large up to 200GB.

# Train subtask 1
# python train.py --ni --config imagenet_256.yml --path_y imagenet --eta 0.85 --deg "sr_bicubic" --deg_scale 4 --sigma_y 0. -i imagenet_sr_bc_4_5 --target_steps 5
# # Train subtask 2
# python train.py --ni --config imagenet_256.yml --path_y imagenet --eta 0.85 --deg "sr_bicubic" --deg_scale 4 --sigma_y 0. -i imagenet_sr_bc_4_5 --target_steps 5 --second_stage
# # Evaluation
# python eval.py --ni --config imagenet_256.yml --path_y imagenet --eta 0.85 --deg "sr_bicubic" --deg_scale 4 --sigma_y 0. -i imagenet_sr_bc_4_5_eval --target_steps 5 --eval_model_name SR_2agent_A2C_5