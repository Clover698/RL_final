import torch
import os
from PIL import Image
def save_image(tensor, output_path, file_name):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    if tensor.dim() == 4:  # [1, C, H, W]
        tensor = tensor.squeeze(0)
    tensor = tensor.clamp(0, 1)  
    tensor = tensor.cpu()  
    image = Image.fromarray((tensor.permute(1, 2, 0).numpy() * 255).astype('uint8'))
    save_path = os.path.join(output_path, file_name)
    image.save(save_path)
    print(f"Image saved at: {save_path}")
    return save_path