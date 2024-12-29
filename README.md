# FOR DDRM
**DATA and Checkpoint**
Download [here](https://drive.google.com/drive/folders/1cSCTaBtnL7OIKXT4SVME88Vtk4uDd_u4)!

Place the checkpoints under the `ddrm/exp/logs/` folder.

CelebA datasets can be found here: [https://www.kaggle.com/datasets/badasstechie/celebahq-resized-256x256](https://www.kaggle.com/datasets/badasstechie/celebahq-resized-256x256). Place the images under the `{your_input_root}/celeba_hq/datasets/` folder.

The models and datasets are placed in the `ddrm/exp/` folder as follows:
```bash
<ddrm/exp> # a folder named by the argument `--exp` given to main.py
├── datasets # all dataset files
│   ├── celeba # all CelebA files
│   ├── imagenet # all ImageNet files
├── logs # contains checkpoints and samples produced during training
│   ├── celeba
│   │   └── celeba_hq.ckpt # the checkpoint file for CelebA-HQ
│   ├── diffusion_models_converted
│   │   └── ema_diffusion_lsun_<category>_model
│   │       └── model-x.ckpt # the checkpoint file saved at the x-th training iteration
│   ├── imagenet # ImageNet checkpoint files
│   │   ├── 256x256_classifier.pt
│   │   ├── 256x256_diffusion.pt
│   │   ├── 256x256_diffusion_uncond.pt
│   │   ├── 512x512_classifier.pt
│   │   └── 512x512_diffusion.pt
├── image_samples # contains generated samples
└── imagenet_val_1k.txt # list of the 1k images used in ImageNet-1K.
```

## Quickstart

Install required packages first:
```bash
pip install -r requirements.txt
```

#### To run the code for both training and evaluate
running the deblurring task:
```bash
bash run_deblur.sh <input_root>
```

running the super-resolution task:
```bash
bash run_sr.sh <input_root>
```

Note that `<inputroot>` refers to the root of the dataset.



### Sampling from the model

The general command to sample from the model is as follows:
```
python main.py --ni --config {CONFIG}.yml --doc {DATASET} --timesteps {STEPS} --eta {ETA} --etaB {ETA_B} --deg {DEGRADATION} --sigma_0 {SIGMA_0} -i {IMAGE_FOLDER}
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



### Images for Demonstration Purposes
CelebA noisy 4x super-resolution:
```
python train.py --ni --config celeba_hq.yml --doc celeba --timesteps 20 --eta 0.85 --etaB 1 --deg sr4 --sigma_0 0.05 -i celeba_hq_sr4_sigma_0.05
```

General content images uniform deblurring:
```
python train.py --ni --config imagenet_256.yml --doc imagenet_ood --timesteps 20 --eta 0.85 --etaB 1 --deg deblur_uni --sigma_0 0.0 -i imagenet_sr4_sigma_0.0
```

Bedroom noisy 4x super-resolution:
```
python train.py --ni --config bedroom.yml --doc bedroom --timesteps 20 --eta 0.85 --etaB 1 --deg sr4 --sigma_0 0.05 -i bedroom_sr4_sigma_0.05
```