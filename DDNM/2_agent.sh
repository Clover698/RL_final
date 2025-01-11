# SR 5
# train ours (1st subtask)
python train.py --ni --config imagenet_256.yml --path_y imagenet --eta 0.85 --deg "sr_bicubic" --deg_scale 4 --sigma_y 0. -i imagenet_sr_bc_4_5_ --target_steps 5
# train ours (2nd subtask)
python train.py --ni --config imagenet_256.yml --path_y imagenet --eta 0.85 --deg "sr_bicubic" --deg_scale 4 --sigma_y 0. -i imagenet_sr_bc_4_5_ --target_steps 5 --second_stage
# eval ours
python eval.py --ni --config imagenet_256.yml --path_y imagenet --eta 0.85 --deg "sr_bicubic" --deg_scale 4 --sigma_y 0. -i imagenet_sr_bc_4_5_eval --target_steps 5 --eval_model_name sr_bicubic_imagenet_2_agents_A2C_5


# SR 10
# train ours (1st subtask)
# python train.py --ni --config imagenet_256.yml --path_y imagenet --eta 0.85 --deg "sr_bicubic" --deg_scale 4 --sigma_y 0. -i imagenet_sr_bc_4_10_ --target_steps 10
# # train ours (2nd subtask)
# python train.py --ni --config imagenet_256.yml --path_y imagenet --eta 0.85 --deg "sr_bicubic" --deg_scale 4 --sigma_y 0. -i imagenet_sr_bc_4_10_ --target_steps 10 --second_stage
# # eval ours
# python eval.py --ni --config imagenet_256.yml --path_y imagenet --eta 0.85 --deg "sr_bicubic" --deg_scale 4 --sigma_y 0. -i imagenet_sr_bc_4_10_eval --target_steps 10 --eval_model_name sr_bicubic_imagenet_2_agents_A2C_10 

# SR 20
# train ours (1st subtask)
# python train.py --ni --config imagenet_256.yml --path_y imagenet --eta 0.85 --deg "sr_bicubic" --deg_scale 4 --sigma_y 0. -i imagenet_sr_bc_4_20_ --target_steps 20
# # train ours (2nd subtask)
# python train.py --ni --config imagenet_256.yml --path_y imagenet --eta 0.85 --deg "sr_bicubic" --deg_scale 4 --sigma_y 0. -i imagenet_sr_bc_4_20_ --target_steps 20 --second_stage
# # eval ours
# python eval.py --ni --config imagenet_256.yml --path_y imagenet --eta 0.85 --deg "sr_bicubic" --deg_scale 4 --sigma_y 0. -i imagenet_sr_bc_4_20_eval --target_steps 20 --eval_model_name sr_bicubic_imagenet_2_agents_A2C_20

# DB 5
# train ours (1st subtask)
# python train.py --ni --config imagenet_256.yml --path_y imagenet --eta 0.85 --deg "deblur_gauss" --sigma_y 0. -i imagenet_deblur_g_5_  --target_steps 5
# # train ours (2nd subtask)
# python train.py --ni --config imagenet_256.yml --path_y imagenet --eta 0.85 --deg "deblur_gauss" --sigma_y 0. -i imagenet_deblur_g_5_  --target_steps 5 --second_stage
# # eval ours
# python eval.py --ni --config imagenet_256.yml --path_y imagenet --eta 0.85 --deg "deblur_gauss" --sigma_y 0. -i imagenet_deblur_g_5_eval --target_steps 5 --eval_model_name deblur_gauss_imagenet_2_agents_A2C_5

# DB 10
# train ours (1st subtask)
# python train.py --ni --config imagenet_256.yml --path_y imagenet --eta 0.85 --deg "deblur_gauss" --sigma_y 0. -i imagenet_deblur_g_10_  --target_steps 10
# # train ours (2nd subtask)
# python train.py --ni --config imagenet_256.yml --path_y imagenet --eta 0.85 --deg "deblur_gauss" --sigma_y 0. -i imagenet_deblur_g_10_  --target_steps 10 --second_stage
# # eval ours
# python eval.py --ni --config imagenet_256.yml --path_y imagenet --eta 0.85 --deg "deblur_gauss" --sigma_y 0. -i imagenet_deblur_g_10_eval --target_steps 10 --eval_model_name deblur_gauss_imagenet_2_agents_A2C_10 

# DB 20
# train ours (1st subtask)
# python train.py --ni --config imagenet_256.yml --path_y imagenet --eta 0.85 --deg "deblur_gauss" --sigma_y 0. -i imagenet_deblur_g_20_  --target_steps 20
# # train ours (2nd subtask)
# python train.py --ni --config imagenet_256.yml --path_y imagenet --eta 0.85 --deg "deblur_gauss" --sigma_y 0. -i imagenet_deblur_g_20_  --target_steps 20 --second_stage
# # eval ours
# python eval.py --ni --config imagenet_256.yml --path_y imagenet --eta 0.85 --deg "deblur_gauss" --sigma_y 0. -i imagenet_deblur_g_20_eval --target_steps 20 --eval_model_name deblur_gauss_imagenet_2_agents_A2C_20
