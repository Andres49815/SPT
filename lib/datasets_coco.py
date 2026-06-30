"""
COCO format segmentation dataset for SAM3 fine-tuning.
Returns image, mask, and bounding box prompt.
"""
import json
import os
import numpy as np
import torch
from PIL import Image, ImageDraw
import torchvision.transforms as transforms


class COCOSegmentationDataset(torch.utils.data.Dataset):
    """
    COCO format segmentation dataset.
     Supports either of these layouts:
     1) Standard COCO-style:
         - data_path/images/{split}/*.jpg
         - data_path/annotations/instances_{split}.json
     2) Split-folder style (e.g. Roboflow export):
         - data_path/{split}/*.jpg
         - data_path/{split}/_annotations.coco.json
    """
    
    def __init__(
        self,
        data_path: str,
        split: str = 'train',
        transform=None,
        input_size: int = 1024,
        num_classes: int = 1,
        mask_type: str = 'binary',
    ):
        """
        Args:
            data_path: Root directory containing 'images' and 'annotations'
            split: 'train', 'val', or 'test'
            transform: Transform to apply to images
            input_size: Target input size for SAM3
            num_classes: Number of segmentation classes
            mask_type: 'binary' or 'multiclass'
        """
        self.data_path = data_path
        self.split = split
        self.input_size = input_size
        self.num_classes = num_classes
        self.mask_type = mask_type
        
        anno_path, image_dir = self._resolve_split_paths(data_path, split)
        with open(anno_path, 'r') as f:
            self.coco_data = json.load(f)
        
        # Build image id to annotations mapping
        self.img_id_to_anns = {}
        for ann in self.coco_data['annotations']:
            img_id = ann['image_id']
            if img_id not in self.img_id_to_anns:
                self.img_id_to_anns[img_id] = []
            self.img_id_to_anns[img_id].append(ann)
        
        # Filter images that have annotations
        self.images = [img for img in self.coco_data['images'] 
                       if img['id'] in self.img_id_to_anns]
        
        self.image_dir = image_dir
        
        # Transform
        if transform is None:
            transform = transforms.Compose([
                transforms.Resize((input_size, input_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
            ])
        self.transform = transform

    def _resolve_split_paths(self, data_path: str, split: str):
        """Resolve annotation and image directories for multiple COCO layouts."""
        split_aliases = [split]
        if split == 'val':
            split_aliases.append('valid')
        if split == 'valid':
            split_aliases.append('val')

        # 1) Split-folder layout: data_path/{split}/_annotations.coco.json
        for split_name in split_aliases:
            anno = os.path.join(data_path, split_name, '_annotations.coco.json')
            img_dir = os.path.join(data_path, split_name)
            if os.path.isfile(anno) and os.path.isdir(img_dir):
                return anno, img_dir

        # 2) Standard COCO-style layout
        for split_name in split_aliases:
            anno = os.path.join(data_path, 'annotations', f'instances_{split_name}.json')
            img_dir = os.path.join(data_path, 'images', split_name)
            if os.path.isfile(anno) and os.path.isdir(img_dir):
                return anno, img_dir

        tried = []
        for split_name in split_aliases:
            tried.append(os.path.join(data_path, split_name, '_annotations.coco.json'))
            tried.append(os.path.join(data_path, 'annotations', f'instances_{split_name}.json'))
        raise FileNotFoundError(
            f'Could not find COCO annotation file for split="{split}". Tried: {tried}'
        )
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        """
        Returns:
            image: (3, H, W) normalized image
            mask: (num_classes, H, W) or (1, H, W) segmentation mask
            box: (4,) normalized bounding box [x1, y1, x2, y2]
        """
        img_info = self.images[idx]
        img_id = img_info['id']
        img_path = os.path.join(self.image_dir, img_info['file_name'])
        if not os.path.isfile(img_path):
            # Some exporters store only the basename in file_name.
            img_path = os.path.join(self.image_dir, os.path.basename(img_info['file_name']))
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        img_w, img_h = image.size
        
        # Get annotations for this image
        annotations = self.img_id_to_anns[img_id]
        
        # Sample one random annotation
        ann = annotations[np.random.randint(0, len(annotations))]
        
        # Extract segmentation mask
        if 'segmentation' in ann:
            seg = ann['segmentation']
            if isinstance(seg, list):  # Polygon format
                mask = self._polygon_to_mask(seg, img_h, img_w)
            else:  # RLE format
                from pycocotools import mask as mask_utils
                mask = mask_utils.decode(seg).astype(np.float32)
        else:
            mask = np.zeros((img_h, img_w), dtype=np.float32)
        
        # Extract bounding box
        bbox = ann['bbox']  # [x, y, w, h]
        x1, y1, w, h = bbox
        x2, y2 = x1 + w, y1 + h
        
        # Normalize bbox to [0, 1]
        box = np.array([x1 / img_w, y1 / img_h, x2 / img_w, y2 / img_h], dtype=np.float32)
        
        # Apply transform to image
        if self.transform is not None:
            image = self.transform(image)
        
        # Resize mask to match input size
        mask = Image.fromarray((mask * 255).astype(np.uint8))
        mask = transforms.Resize((self.input_size, self.input_size),
                     interpolation=transforms.InterpolationMode.NEAREST)(mask)
        mask = np.array(mask, dtype=np.float32) / 255.0
        
        # Convert mask to tensor and add channel dimension
        mask = torch.from_numpy(mask).unsqueeze(0)  # (1, H, W)
        
        # For multiclass, replicate mask across classes
        if self.mask_type == 'multiclass':
            mask = mask.repeat(self.num_classes, 1, 1)
        
        box = torch.from_numpy(box).float()
        
        return image, mask, box
    
    def _polygon_to_mask(self, polygons, height, width):
        """Convert polygon annotation to binary mask."""
        from pycocotools import mask as mask_utils
        
        mask = np.zeros((height, width), dtype=np.uint8)
        for polygon in polygons:
            polygon = np.array(polygon, dtype=np.int32).reshape(-1, 2)
            # Draw polygon (simplified approach)
            from PIL import ImageDraw
            img = Image.new('L', (width, height), 0)
            draw = ImageDraw.Draw(img)
            draw.polygon([tuple(p) for p in polygon], fill=1)
            mask = np.maximum(mask, np.array(img, dtype=np.uint8))
        
        return mask.astype(np.float32)


def build_coco_segmentation_dataset(
    is_train: bool,
    data_path: str,
    input_size: int = 1024,
    num_classes: int = 1,
    mask_type: str = 'binary',
):
    """
    Build COCO segmentation dataset.
    
    Args:
        is_train: Whether to use train or val split
        data_path: Path to COCO dataset root
        input_size: SAM3 input size
        num_classes: Number of classes
        mask_type: 'binary' or 'multiclass'
    
    Returns:
        COCOSegmentationDataset instance
    """
    split = 'train' if is_train else 'valid'
    dataset = COCOSegmentationDataset(
        data_path=data_path,
        split=split,
        input_size=input_size,
        num_classes=num_classes,
        mask_type=mask_type,
    )
    return dataset
