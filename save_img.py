import torch
import os
from PIL import Image

def save_image(tensor, output_path, file_name):
    """
    將 PyTorch 張量保存為圖像文件。

    Args:
        tensor (torch.Tensor): 輸入的圖像張量，形狀為 [C, H, W] 或 [1, C, H, W]。
        output_path (str): 圖像保存的目錄。
        file_name (str): 保存的文件名，例如 "output_image.png"。

    Returns:
        str: 完整的保存路徑。
    """
    # 確保輸出路徑存在
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # 將張量轉換為 PIL 圖像
    if tensor.dim() == 4:  # 如果輸入是 [1, C, H, W] 格式
        tensor = tensor.squeeze(0)
    tensor = tensor.clamp(0, 1)  # 將值限制在 [0, 1] 範圍內
    tensor = tensor.cpu()  # 確保張量在 CPU 上
    image = Image.fromarray((tensor.permute(1, 2, 0).numpy() * 255).astype('uint8'))

    # 保存圖像
    save_path = os.path.join(output_path, file_name)
    image.save(save_path)

    print(f"Image saved at: {save_path}")
    return save_path
