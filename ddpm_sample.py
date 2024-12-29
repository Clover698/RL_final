import argparse, os, torch
from tqdm import tqdm
import accelerate
import torch_pruning as tp
import numpy as np
import random
parser = argparse.ArgumentParser()
parser.add_argument("--total_samples", type=int, default=50000)
parser.add_argument("--batch_size", type=int, default=100)
parser.add_argument("--output_dir", type=str, default="samples")
parser.add_argument("--model_path", type=str, default="samples")
parser.add_argument("--ddim_steps", type=int, default=100)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--skip_type", type=str, default="uniform")

args = parser.parse_args()

if __name__ == "__main__":
    generator = torch.Generator(device='cpu').manual_seed(232)
    seed = np.random.randint(232)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.random.manual_seed(232)
    os.makedirs(args.output_dir, exist_ok=True)
    # pruned model
    accelerator = accelerate.Accelerator()

    if os.path.isdir(args.model_path):
        from diffusers_old import DDIMPipeline, DDIMScheduler, UNet2DModel
        print("Loading model from {}".format(args.model_path))
        subfolder = 'unet' if os.path.isdir(os.path.join(args.model_path, 'unet')) else None
        unet = UNet2DModel.from_pretrained(args.model_path, subfolder=subfolder).eval()
        pipeline = DDIMPipeline(
            unet=unet,
            scheduler=DDIMScheduler.from_pretrained(args.model_path, subfolder="scheduler")
        )
    # standard model
    else:  
        from diffusers import DDIMPipeline, DDIMScheduler, UNet2DModel
        print("Loading pretrained model from {}".format(args.model_path))
        pipeline = DDIMPipeline.from_pretrained(args.model_path)

    pipeline.scheduler.skip_type = args.skip_type
    pipeline.to(accelerator.device)
    
    # Create subfolders for each process
    save_sub_dir = os.path.join(args.output_dir, 'process_{}'.format(accelerator.process_index))
    os.makedirs(save_sub_dir, exist_ok=True)
    generator = torch.Generator(device=pipeline.device).manual_seed(args.seed+accelerator.process_index)

    # Set up progress bar
    if not accelerator.is_main_process:
        pipeline.set_progress_bar_config(disable=True)
    
    # Sampling
    accelerator.wait_for_everyone()
    with torch.no_grad():
        # num_batches of each process
        num_batches = (args.total_samples) // (args.batch_size * accelerator.num_processes)
        if accelerator.is_main_process:
            print("Samping {}x{}={} images with {} process(es)".format(num_batches*args.batch_size, accelerator.num_processes, num_batches*accelerator.num_processes*args.batch_size, accelerator.num_processes))
        for i in tqdm(range(num_batches), disable=not accelerator.is_main_process):
            images = pipeline(batch_size=args.batch_size, num_inference_steps=args.ddim_steps, generator=generator).images
            for j, image in enumerate(images):
                filename = os.path.join(save_sub_dir, f"{i * args.batch_size + j}.png")
                image.save(filename)

    # Finished
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        accelerator.print(f"Saved {num_batches*accelerator.num_processes*args.batch_size} samples to {args.output_dir}")
    #accelerator.end_training()
    
