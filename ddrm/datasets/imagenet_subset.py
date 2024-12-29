import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image

class CenterCropLongEdge(object):
    """Crops the given PIL Image on the long edge.
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
    """

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be cropped.
        Returns:
            PIL Image: Cropped image.
        """
        return transforms.functional.center_crop(img, min(img.size))

    def __repr__(self):
        return self.__class__.__name__

def pil_loader(path):
    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)

def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)

# class ImageDataset(data.Dataset):

#     def __init__(self,
#                  root_dir,
#                  meta_file,
#                  transform=None,
#                  image_size=128,
#                  normalize=True):
#         self.root_dir = root_dir
#         if transform is not None:
#             self.transform = transform
#         else:
#             norm_mean = [0.5, 0.5, 0.5]
#             norm_std = [0.5, 0.5, 0.5]
#             if normalize:
#                 self.transform = transforms.Compose([
#                     CenterCropLongEdge(),
#                     transforms.Resize(image_size),
#                     transforms.ToTensor(),
#                     transforms.Normalize(norm_mean, norm_std)
#                 ])
#             else:
#                 self.transform = transforms.Compose([
#                     CenterCropLongEdge(),
#                     transforms.Resize(image_size),
#                     transforms.ToTensor()
#                 ])
#         with open(meta_file) as f:
#             lines = f.readlines()
#         print("building dataset from %s" % meta_file)
#         self.num = len(lines)
#         self.metas = []
#         self.classifier = None
#         suffix =  ".jpeg"
#         for line in lines:
#             line_split = line.rstrip().split()
#             if len(line_split) == 2:
#                 self.metas.append((line_split[0] + suffix, int(line_split[1])))
#             else:
#                 self.metas.append((line_split[0] + suffix, -1))
#         print("read meta done")

#     def __len__(self):
#         return self.num

#     def __getitem__(self, idx):
#         filename = self.root_dir + '/' + self.metas[idx][0]
#         cls = self.metas[idx][1]
#         img = default_loader(filename)

#         # transform
#         if self.transform is not None:
#             img = self.transform(img)

#         return img, cls #, self.metas[idx][0]
    
import os
class ImageDataset(data.Dataset):

    def __init__(self, root_dir, transform=None, image_size=256, normalize=True):
        """
        Args:
            root_dir (str): 根目錄，包含圖像類別子文件夾。
            transform (callable, optional): 圖像預處理操作。
            image_size (int, optional): 圖像大小，用於 resize。
            normalize (bool, optional): 是否對圖像進行標準化。
        """
        self.root_dir = root_dir
        self.classes = sorted(os.listdir(root_dir))  # 假設子文件夾名稱對應類別
        self.samples = []
        for cls_idx, cls_name in enumerate(self.classes):
            cls_path = os.path.join(root_dir, cls_name)
            if os.path.isdir(cls_path):
                for img_name in os.listdir(cls_path):
                    self.samples.append((os.path.join(cls_path, img_name), cls_idx))

        if transform is not None:
            self.transform = transform
        else:
            norm_mean = [0.5, 0.5, 0.5]
            norm_std = [0.5, 0.5, 0.5]
            if normalize:
                self.transform = transforms.Compose([
                    transforms.Resize(image_size),
                    transforms.CenterCrop(image_size),
                    transforms.ToTensor(),
                    transforms.Normalize(norm_mean, norm_std)
                ])
            else:
                self.transform = transforms.Compose([
                    transforms.Resize(image_size),
                    transforms.CenterCrop(image_size),
                    transforms.ToTensor()
                ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        filepath, cls = self.samples[idx]
        img = Image.open(filepath).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        return img, cls
