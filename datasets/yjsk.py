"""
YJSK dataset which returns image_id for evaluation.
Similar to COCO dataset structure.
"""
from pathlib import Path
import torch
import torch.utils.data
import torchvision
import datasets.transforms as T
from .coco import make_coco_transforms_square_div_64,make_coco_transforms


class YJSKDetection(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, ann_file, transforms):
        super(YJSKDetection, self).__init__(img_folder, ann_file)
        self._transforms = transforms
        self.prepare = ConvertYJSK()

    def __getitem__(self, idx):
        img, target = super(YJSKDetection, self).__getitem__(idx)
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}
        img, target = self.prepare(img, target)
        if self._transforms is not None:
            img, target = self._transforms(img, target)
            # print('image shape:',img.shape)
        return img, target


class ConvertYJSK(object):
    def __call__(self, image, target):
        w, h = image.size
        # print('=======================================',image.size)

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]
        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]

        boxes = [obj["bbox"] for obj in anno]
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]  # convert xywh to xyxy
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        target["image_id"] = image_id
        target["area"] = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        target["iscrowd"] = torch.zeros(len(boxes), dtype=torch.int64)
        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])
        # print("====================",target)

        
        return image, target


def build(image_set, args):
    root = Path(args.yjsk_path)
    assert root.exists(), f'provided YJSK path {root} does not exist'
    
    PATHS = {
        "train": (root / "train", root / "annotations" / "instances_train.json"),
        "val": (root / "val", root / "annotations" / "instances_val.json"),
        "test": (root / "test", root / "annotations" / "instances_test.json")
    }
    
    img_folder, ann_file = PATHS[image_set]
    
    try:
        square_resize_div_64 = args.square_resize_div_64
    except:
        square_resize_div_64 = False

    dataset = YJSKDetection(
        img_folder, 
        ann_file,
        transforms=make_coco_transforms_square_div_64(image_set) if square_resize_div_64 else make_coco_transforms(image_set)
    )

    
    return dataset
