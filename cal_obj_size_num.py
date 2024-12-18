import os
import json
import numpy as np
from tqdm import tqdm

def count_objects_by_size_ratio(json_path):
    """统计数据集中大中小物体的数量
    大物体: 面积 > 图像面积的1/32
    中等物体: 图像面积的1/256 <= 面积 <= 图像面积的1/32
    小物体: 面积 < 图像面积的1/256
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # 建立图像id到尺寸的映射
    image_sizes = {img['id']: (img['height'], img['width']) for img in data['images']}
    
    small, medium, large = 0, 0, 0
    categories = {cat['id']: cat['name'] for cat in data['categories']}
    category_stats = {cat_id: {'small': 0, 'medium': 0, 'large': 0} 
                     for cat_id in categories.keys()}
    
    for ann in tqdm(data['annotations']):
        bbox = ann['bbox']  # [x, y, w, h]
        area = bbox[2] * bbox[3]  # w * h
        
        # 获取对应图像的尺寸
        img_h, img_w = image_sizes[ann['image_id']]
        img_area = img_h * img_w
        
        # 计算面积比例
        area_ratio = area / img_area
        
        # 统计不同大小物体的数量
        if area_ratio < 1/256:
            small += 1
            category_stats[ann['category_id']]['small'] += 1
        elif area_ratio <= 1/32:
            medium += 1
            category_stats[ann['category_id']]['medium'] += 1
        else:
            large += 1
            category_stats[ann['category_id']]['large'] += 1
    total = small + medium + large
    print(f"小物体 (<1/256): {small} ({small/total*100:.1f}%)")
    print(f"中等物体 (1/256~1/32): {medium} ({medium/total*100:.1f}%)")
    print(f"大物体 (>1/32): {large} ({large/total*100:.1f}%)")
    print(f"总物体数量: {small + medium + large}")
    
    print("\n各类别统计:")
    for cat_id, cat_name in categories.items():
        stats = category_stats[cat_id]
        total = sum(stats.values())
        if total > 0:
            print(f"\n{cat_name}:")
            print(f"小物体: {stats['small']} ({stats['small']/total*100:.1f}%)")
            print(f"中等物体: {stats['medium']} ({stats['medium']/total*100:.1f}%)")
            print(f"大物体: {stats['large']} ({stats['large']/total*100:.1f}%)")
            print(f"总数: {total}")


def count_objects_by_size_ratio_total(json_path):
    """统计数据集中大中小物体的总数量"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # 建立图像id到尺寸的映射
    image_sizes = {img['id']: (img['height'], img['width']) for img in data['images']}
    
    small, medium, large = 0, 0, 0
    
    for ann in tqdm(data['annotations']):
        bbox = ann['bbox']  # [x, y, w, h]
        area = bbox[2] * bbox[3]  # w * h
        
        img_h, img_w = image_sizes[ann['image_id']]
        img_area = img_h * img_w
        area_ratio = area / img_area
        
        if area_ratio < 1/256:
            small += 1
        elif area_ratio <= 1/32:
            medium += 1
        else:
            large += 1
    
    total = small + medium + large
    print(f"\n总统计:")
    print(f"小物体 (<1/256): {small} ({small/total*100:.1f}%)")
    print(f"中等物体 (1/256~1/32): {medium} ({medium/total*100:.1f}%)")
    print(f"大物体 (>1/32): {large} ({large/total*100:.1f}%)")
    print(f"总物体数量: {total}")

if __name__ == "__main__":
    # 替换为你的数据集路径
    json_path = "./yjsk/annotations/instances_test.json"  
    count_objects_by_size_ratio(json_path)
    # count_objects_by_size_ratio_total(json_path)
