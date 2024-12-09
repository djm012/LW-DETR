import os
from pathlib import Path
import torch
import torch.utils.data
import torchvision
from PIL import Image
import json
from coco import make_coco_transforms_square_div_64
import datasets.transforms as T

class YJSKDetection(torch.utils.data.Dataset):
    def __init__(self, img_folder, transforms=None):
        self.img_folder = img_folder
        self._transforms = transforms
        
        # 获取所有图片文件
        self.imgs = list(sorted(os.listdir(img_folder)))
        
        # 读取标注文件
        ann_file = os.path.join(os.path.dirname(img_folder), "labels", "instances.json")
        with open(ann_file, 'r') as f:
            self.annotations = json.load(f)
            
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_folder, self.imgs[idx])
        img = Image.open(img_path).convert('RGB')
        
        # 获取对应的标注
        image_id = int(self.imgs[idx].split('.')[0])
        target = self.annotations[str(image_id)]
        
        w, h = img.size
        boxes = torch.as_tensor(target['boxes'], dtype=torch.float32)
        boxes[:, 2:] += boxes[:, :2]  # convert to xywh
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)
        
        classes = torch.tensor(target['labels'], dtype=torch.int64)
        
        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        target["image_id"] = torch.tensor([image_id])
        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])
        
        if self._transforms is not None:
            img, target = self._transforms(img, target)
            
        return img, target

    def __len__(self):
        return len(self.imgs)

def build(image_set, args):
    root = Path(args.yjsk_path)
    assert root.exists(), f'provided YJSK path {root} does not exist'
    
    PATHS = {
        "train": root / "images/train",
        "val": root / "images/val",
        "test": root / "images/test"
    }
    
    img_folder = PATHS[image_set]
    
    dataset = YJSKDetection(
        img_folder,
        transforms=make_coco_transforms_square_div_64(image_set)
    )
    
    return dataset
