# Please modify the dataset path of imagenet in datasets/__init__.py (line 176). You can download the dataset from http://image-net.org/download-images. Note the dataset is large up to 200GB.

# Train subtask 1
python train.py --ni --config imagenet_256.yml --path_y imagenet --eta 0.85 --deg "sr_bicubic" --deg_scale 4 --sigma_y 0. -i imagenet_sr_bc_4_5 --target_steps 5
# Train subtask 2
python train.py --ni --config imagenet_256.yml --path_y imagenet --eta 0.85 --deg "sr_bicubic" --deg_scale 4 --sigma_y 0. -i imagenet_sr_bc_4_5 --target_steps 5 --second_stage
# Evaluation
python eval.py --ni --config imagenet_256.yml --path_y imagenet --eta 0.85 --deg "sr_bicubic" --deg_scale 4 --sigma_y 0. -i imagenet_sr_bc_4_5_eval --target_steps 5 --eval_model_name SR_2agent_A2C_5