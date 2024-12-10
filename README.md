## Quickstart

Install required packages first:
```bash
pip install -r requirements.txt
```

Please download the CIFAR10 model via wget and put it at `model/ddpm_ema_cifar10`:
```bash
wget https://github.com/VainF/Diff-Pruning/releases/download/v0.0.1/ddpm_ema_cifar10.zip
```
PS: Note this model is only supported by old diffusers.

To train RL, run:
```python
python train.py
```

To evaluate FID, run:
```bash
bash eval_fid.sh
```
PS. Please modify `--save_path` in eval_fid.sh, this means the path of the RL model produced by `train.py`.

(Note all these codes may not be 100% accurate)

## Experimental Results - Generation

### CIFAR 10 (Generation)
| | T=5 | T=10 | T=20 | T=100|
| --- | --- | --- | --- | --- |
| DDIM |68.28|20.76|11.46|5.71|
| RL   |34.67|18.13|11.02|-|

Thresholds of sparse reward are set 0.75, 0.89, 0.93 for T of 5, 10, 20.
The setting of thresholds impacts a lot. For example, experiment of T=5 with threshold of 0.8 only gets FID of 66.33.
For PPO, FID under T=10 is 18.26.

### LSUN-Church (Generation)
| | T=5 | T=10 | T=20 | T=100|
| --- | --- | --- | --- | --- |
| DDIM |49.84|19.10 |12.04|10.55|
| RL   |52.97|21.24|12.60|-|

Thresholds of sparse reward are set 0.55, 0.76, 0.89 for T of 5, 10, 20.
The results of RL are worse than DDIM, which are likely caused by the discrepancy between FID and SSIM.
Moreover, prior work has not implemented on high-resolution (256x256) images, which are more difficult tasks.

## Experimental Results - DDNM 

### PSNR (ImageNet1K) (SR) 
|               | T=5  | T=10 | T=20 | T=100 |
|---------------|------|------|------|-------|
| DDNM          | 26.82| 26.95| 27.12| 27.46 |
| RL            | 26.84| 26.94| 27.10| -     |
| subtask 1 (continuous) | 26.89|      |      |       |
| subtask 1 (discrete 10) | 26.86|      |      |       |
| subtask 1 (discrete 20) | 26.90|      |      |       |
| subtask 2 (continuous) | 26.84|      |      |       |
| Combined (continuous) | 26.88|      |      |       |
| Combined (discrete 20) | 26.89|      |      |       |
| Combined (D/S) | 26.91|      |      |       |
| 2 agents      | 26.93|      |      |       |



### SSIM (ImageNet1K) (SR) 
| | T=5 | T=10 | T=20 | T=100|
| --- | --- | --- | --- | --- |
| DDNM |0.880|0.881|0.884|0.890|
| RL   |0.880|0.882|0.884|-|
|subtask 1(continuous)|0.881||||
|subtask 1(discrete10)|0.880||||
|subtask 1(discrete20)|0.881||||
|subtask 2(continuous)|0.881||||
|Combined(continuous)|0.881||||
|Combined(discrete20)|0.881||||
|Combined(D/S)|0.881||||
|2 agents|0.882||||

### PSNR (ImageNet1K) (DB) 
| | T=5 | T=10 | T=20 | T=100|
| --- | --- | --- | --- | --- |
| DDNM |40.57|42.02|43.21|45.13|
| RL   |42.62|||-|
|subtask 1(continuous)|44.00||||
|subtask 1(discrete10)|42.97||||
|subtask 1(discrete20)|44.14||||
|subtask 2(continuous)|43.79||||
|Combined(continuous)|43.97||||
|Combined(discrete20)|43.46||||
|Combined(D/S)|44.06||||
|2 agents|44.38||||
|subtask 1(continuous)(PPO)|42.85||||
|subtask 1(discrete20)(PPO)|43.67||||
|subtask 2(continuous)|42.95||||
|subtask 2(discrete20)|42.52||||


### SSIM (ImageNet1K) (DB) 
| | T=5 | T=10 | T=20 | T=100|
| --- | --- | --- | --- | --- |
| DDNM |0.990|0.992|0.993|0.995|
| RL   |0.993|||-|
|subtask 1(continuous)|0.994||||
|subtask 1(discrete10)|0.993||||
|subtask 1(discrete20)|0.994||||
|subtask 2(continuous)|0.994||||
|Combined(continuous)|0.994||||
|Combined(discrete20)|0.994||||
|Combined(D/S)|0.994||||
|2 agents|0.994||||
|subtask 1(continuous)(PPO)|0.993||||
|subtask 1(discrete20)(PPO)|0.994||||
|subtask 2(continuous)|0.993||||
|subtask 2(discrete20)|0.993||||


### PSNR (CelebaHQ1K) (SR) 
| | T=5 | T=10 | T=20 | T=100|
| --- | --- | --- | --- | --- |
| DDNM |31.76|31.79|31.72|31.71|
| RL   | | | |-|

### SSIM (CelebaHQ1K) (SR) 
| | T=5 | T=10 | T=20 | T=100|
| --- | --- | --- | --- | --- |
| DDNM |0.952|0.952|0.951|0.950|
| RL   | | | |-|

### PSNR (CelebaHQ1K) (DB) 
| | T=5 | T=10 | T=20 | T=100|
| --- | --- | --- | --- | --- |
| DDNM |48.71|50.11|51.65|55.82|
| RL   | | | |-|

### SSIM (CelebaHQ1K) (DB) 
| | T=5 | T=10 | T=20 | T=100|
| --- | --- | --- | --- | --- |
| DDNM |0.998|0.999|0.999|1.000|
| RL   | | | |-|
