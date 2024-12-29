# Adaptive Sampling on Diffusion Model for Low-Level Vision Tasks

## Environment

Install required packages first:

```bash
pip install -r requirements.txt
```

## Data and checkpoint preperation

For celeba checkpoint, please download [here](https://drive.google.com/drive/folders/1cSCTaBtnL7OIKXT4SVME88Vtk4uDd_u4)!

The models and datasets are placed in the `ddrm/exp/` folder as follows:

```bash
<ddrm/exp> # a folder named by the argument `--exp` given to main.py
├── datasets # all dataset files
│   ├── celeba # all CelebA files
│   └── imagenet # all ImageNet files
├── logs # contains checkpoints and samples produced during training
│   ├── celeba
│   │   └── celeba_hq.ckpt # the checkpoint file for CelebA-HQ
│   ├── imagenet # ImageNet checkpoint files
│   │   └── 512x512_diffusion.pt
└── imagenet_val_1k.txt # list of the 1k images used in ImageNet-1K.
```

## Training from the model

The general command to sample from the model is as follows:

```
python train.py --ni --config {CONFIG}.yml --doc {DATASET} --timesteps {STEPS} --eta {ETA} --etaB {ETA_B} --deg {DEGRADATION} --sigma_0 {SIGMA_0} --second_stage --target_steps {TARGET_STEPS} --input_root {INPUT_ROOT}
```

where the following are options

- `ETA` is the eta hyperparameter in the paper. (default: `0.85`)
- `ETA_B` is the eta_b hyperparameter in the paper. (default: `1`)
- `STEPS` controls how many timesteps used in the process.
- `DEGREDATION` is the type of degredation allowed. (One of: `cs2`, `cs4`, `inp`, `inp_lolcat`, `inp_lorem`, `deno`, `deblur_uni`, `deblur_gauss`, `deblur_aniso`, `sr2`, `sr4`, `sr8`, `sr16`, `sr_bicubic4`, `sr_bicubic8`, `sr_bicubic16` `color`)
- `SIGMA_0` is the noise observed in y.
- `CONFIG` is the name of the config file (see `configs/` for a list), including hyperparameters such as batch size and network architectures.
- `DATASET` is the name of the dataset used, to determine where the checkpoint file is found.
- `IMAGE_FOLDER` is the name of the folder the resulting images will be placed in (default: `images`)
- `SECOND_STAGE` is the flag used to train the second agent. If it is not specified, the first agent will be trained by default.
- `INPUT_ROOT` is the root directory in the script.

## Images for Demonstration Purposes

CelebA Noisy 4x Super-Resolution: Target Step 5

```
python train.py --ni --config celeba_hq.yml --doc celeba --timesteps 20 --eta 0.85 --etaB 1 --deg sr4 --sigma_0 0.05 -i celeba_hq_sr4_sigma_0.05 --target_steps 5
```

CelebA Noisy uniform deblurring: Target Step 5

```
python train.py --ni --config celeba_hq.yml --doc celeba --timesteps 20 --eta 0.85 --etaB 1 --deg deblur_uni --sigma_0 0.0 -i celeba_hq_sr4_sigma_0.05 --target_steps 5

```

## Full Pipeline

To run the pipeline for training the first and second agents, as well as evaluation at target steps 5, 10, and 20, simply execute the following scripts:

CelebA Noisy 4x Super-Resolution: Target Step 5, 10, 20

```bash
./run_sr4.sh {INPUT_ROOT}
```

CelebA Noisy 4x uniform deblurring: Target Step 5, 10, 20

```bash
./run_deblur.sh {INPUT_ROOT}
```

## Experimental Results

### Comparison of PSNR and SSIM on 4x Super-Resolution on CelebA-HQ_256

| Method                    | **Step 5** | **Step 5** | **Step 10** | **Step 10** | **Step 20** | **Step 20** |
| ------------------------- | ---------- | ---------- | ----------- | ----------- | ----------- | ----------- |
|                           | PSNR↑      | SSIM↑      | PSNR↑       | SSIM↑       | PSNR↑       | SSIM↑       |
| **DDRM**                  | **28.524** | **0.892**  | **28.865**  | **0.899**   | **29.130**  | **0.905**   |
| DDRM + RS-DDIM (baseline) | 27.380     | 0.867      | 27.640      | 0.870       | 27.860      | 0.883       |
| DDRM + ours               | 27.650     | 0.876      | 27.980      | 0.881       | 27.720      | 0.875       |
| **DDNM**                  | 31.77      | **0.952**  | 31.80       | **0.952**   | 31.72       | 0.951       |
| DDNM + RS-DDIM (baseline) | 31.77      | **0.952**  | 31.79       | **0.952**   | 31.71       | 0.951       |
| DDNM + ours               | **31.80**  | **0.952**  | **31.85**   | **0.952**   | **31.90**   | **0.953**   |

### Comparison of PSNR and SSIM on Deblurring on CelebA-HQ_256

| Method                    | **Step 5** | **Step 5** | **Step 10** | **Step 10** | **Step 20** | **Step 20** |
| ------------------------- | ---------- | ---------- | ----------- | ----------- | ----------- | ----------- |
|                           | PSNR↑      | SSIM↑      | PSNR↑       | SSIM↑       | PSNR↑       | SSIM↑       |
| **DDRM**                  | 41.331     | 0.991      | 42.420      | 0.993       | 43.458      | 0.995       |
| DDRM + RS-DDIM (baseline) | 42.878     | 0.994      | 44.337      | 0.995       | **45.836**  | **0.997**   |
| DDRM + ours               | **44.869** | **0.996**  | **45.311**  | **0.996**   | 44.138      | 0.995       |
| **DDNM**                  | 48.72      | 0.998      | 50.1        | 0.999       | 51.64       | 0.999       |
| DDNM + RS-DDIM (baseline) | 52.16      | 0.999      | 52.75       | 0.999       | 54.61       | **1.000**   |
| DDNM + ours               | **55.48**  | **1.000**  | **54.71**   | **1.000**   | **54.70**   | **1.000**   |
