CUDA_VISIBLE_DEVICES=1 python ddpm_sample.py \
--output_dir run/sample/ddim_church_pretrained \
--model_path google/ddpm-ema-church-256 \
--batch_size 32 \

# Compute the FID score of sampled images
CUDA_VISIBLE_DEVICES=1 python fid_score.py run/sample/ddim_church_pretrained run/fid_stats_church.npz --device cuda:0 --batch-size 256

