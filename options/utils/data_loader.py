import os
import json
import math
import torch
import random
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T
from shapely import wkt

# Maps the xBD "subtype" to an integer label
DAMAGE_LABELS = {
    "no-damage": 0,
    "minor-damage": 1,
    "major-damage": 2,
    "destroyed": 3
    # If there's "unclassified", you can decide to skip or map it to 0
}

class XBDPatchDataset(Dataset):
    """
    On-the-fly dataset that:
      - Scans all disasters under data/xBD
      - For each post-disaster JSON, enumerates building polygons
      - Computes bounding boxes, then center-crops the pre & post images
      - Returns (pre_patch, post_patch, label)
    """
    def __init__(self,
                 root_dir,
                 pre_crop_size=128,
                 post_crop_size=224,
                 use_xy=True,
                 max_samples=None,
                 flat_structure=False,
                 augment=True):
        """
        :param root_dir: Path to "data/xBD" (containing subfolders for each disaster)
        :param pre_crop_size: Patch size for pre-disaster images (shallow CNN)
        :param post_crop_size: Patch size for post-disaster images (ResNet50)
        :param use_xy: If True, use 'xy' coords from JSON; else use 'lng_lat'
        :param max_samples: If not None, limit total building samples to this number
        :param flat_structure: If True, assumes images are in a flat directory structure
        :param augment: If True, apply data augmentation
        """
        super().__init__()
        self.root_dir = root_dir
        self.pre_crop_size = pre_crop_size
        self.post_crop_size = post_crop_size
        self.coord_key = "xy" if use_xy else "lng_lat"
        self.max_samples = max_samples
        self.flat_structure = flat_structure
        self.augment = augment
        
        # Check if we're using '/test' directory structure which has images and labels subdirs
        if os.path.basename(os.path.normpath(root_dir)) == 'test' and os.path.isdir(os.path.join(root_dir, 'images')) and os.path.isdir(os.path.join(root_dir, 'labels')):
            self.test_structure = True
        else:
            self.test_structure = False

        # We'll define transforms for each branch
        if augment:
            self.pre_transform = T.Compose([
                T.Resize((pre_crop_size, pre_crop_size)),
                T.RandomHorizontalFlip(),
                T.RandomVerticalFlip(),
                T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
            ])
            self.post_transform = T.Compose([
                T.Resize((post_crop_size, post_crop_size)),
                T.RandomHorizontalFlip(),
                T.RandomVerticalFlip(),
                T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
            ])
        else:
            self.pre_transform = T.Compose([
                T.Resize((pre_crop_size, pre_crop_size)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
            ])
            self.post_transform = T.Compose([
                T.Resize((post_crop_size, post_crop_size)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
            ])

        # Collect a list of all building samples (pre_path, post_path, bounding_box, label)
        self.samples = self._gather_samples()

        # If user wants to limit the number of samples
        if self.max_samples is not None and len(self.samples) > self.max_samples:
            self.samples = random.sample(self.samples, self.max_samples)

    def _gather_samples(self):
        """
        Iterates over each disaster folder, each *_post_disaster.json,
        enumerates polygons, and collects:
          - pre_image_path
          - post_image_path
          - bounding box (x1, y1, x2, y2)
          - label (0..3)
        Returns a list of dicts, one per building.
        """
        samples = []
        
        if self.flat_structure:
            # Flat structure: images directly in root_dir
            # For flat structure, we pair pre and post images directly
            files = os.listdir(self.root_dir)
            pre_images = [f for f in files if f.endswith("_pre_disaster.png")]
            
            for pre_image_name in pre_images:
                base_id = pre_image_name.replace("_pre_disaster.png", "")
                post_image_name = base_id + "_post_disaster.png"
                
                # Ensure post-disaster image exists
                if post_image_name not in files:
                    continue
                
                pre_image_path = os.path.join(self.root_dir, pre_image_name)
                post_image_path = os.path.join(self.root_dir, post_image_name)
                
                # In flat structure mode, we take the whole image and use a default label
                # This assumes you're using the dataset for pre/post comparison without labels
                try:
                    with Image.open(pre_image_path) as img:
                        width, height = img.size
                    
                    # Use the entire image as the bbox
                    samples.append({
                        "pre_img": pre_image_path,
                        "post_img": post_image_path,
                        "bbox": (0, 0, width, height),
                        "label": 0  # Default label, can be modified if needed
                    })
                except Exception as e:
                    print(f"Error processing {pre_image_path}: {e}")
                    continue
        elif self.test_structure:
            # Test structure: has 'images' and 'labels' subdirectories
            images_dir = os.path.join(self.root_dir, "images")
            labels_dir = os.path.join(self.root_dir, "labels")
            
            # Get all image files
            image_files = [f for f in os.listdir(images_dir) if f.endswith(".png")]
            pre_images = [f for f in image_files if f.endswith("_pre_disaster.png")]
            
            for pre_image_name in pre_images:
                base_id = pre_image_name.replace("_pre_disaster.png", "")
                post_image_name = base_id + "_post_disaster.png"
                pre_json_name = base_id + "_pre_disaster.json"
                post_json_name = base_id + "_post_disaster.json"
                
                # Ensure post-disaster image exists
                if post_image_name not in image_files:
                    continue
                
                pre_image_path = os.path.join(images_dir, pre_image_name)
                post_image_path = os.path.join(images_dir, post_image_name)
                post_json_path = os.path.join(labels_dir, post_json_name)
                pre_json_path = os.path.join(labels_dir, pre_json_name)
                
                # Check if all files exist
                if not os.path.isfile(post_json_path) or not os.path.isfile(pre_json_path):
                    continue
                
                # Load JSON
                try:
                    with open(post_json_path, 'r') as f:
                        post_data = json.load(f)
                    
                    # Parse features
                    feats = post_data.get("features", {}).get(self.coord_key, [])
                    
                    # If no features, use the entire image with a default label
                    if not feats:
                        with Image.open(pre_image_path) as img:
                            width, height = img.size
                        
                        samples.append({
                            "pre_img": pre_image_path,
                            "post_img": post_image_path,
                            "bbox": (0, 0, width, height),
                            "label": 0  # Default label
                        })
                        continue
                    
                    for feat in feats:
                        damage_type = feat.get("properties", {}).get("subtype", "").lower()
                        if damage_type not in DAMAGE_LABELS:
                            # Skip if "unclassified" or unknown
                            continue
                        label = DAMAGE_LABELS[damage_type]
                        
                        # Parse polygon
                        wkt_str = feat.get("wkt", None)
                        if wkt_str is None:
                            continue
                        polygon = wkt.loads(wkt_str)
                        # Compute bounding box (minx, miny, maxx, maxy)
                        minx, miny, maxx, maxy = polygon.bounds
                        
                        # Store sample
                        samples.append({
                            "pre_img": pre_image_path,
                            "post_img": post_image_path,
                            "bbox": (minx, miny, maxx, maxy),
                            "label": label
                        })
                
                except Exception as e:
                    print(f"Error processing {post_json_path}: {e}")
                    continue
        else:
            # Regular structure: disaster subfolders with images and labels
            disasters = [
                d for d in os.listdir(self.root_dir)
                if os.path.isdir(os.path.join(self.root_dir, d))
                   and d.lower() != "spacenet_gt"
            ]

            for disaster in disasters:
                disaster_dir = os.path.join(self.root_dir, disaster)
                images_dir = os.path.join(disaster_dir, "images")
                labels_dir = os.path.join(disaster_dir, "labels")

                # skip if no images/labels folder
                if not (os.path.isdir(images_dir) and os.path.isdir(labels_dir)):
                    continue

                # get all post-disaster JSON files
                label_files = [
                    f for f in os.listdir(labels_dir)
                    if f.endswith("_post_disaster.json")
                ]
                for label_file in label_files:
                    base_id = label_file.replace("_post_disaster.json", "")
                    post_image_name = base_id + "_post_disaster.png"
                    pre_image_name  = base_id + "_pre_disaster.png"

                    post_json_path = os.path.join(labels_dir, label_file)
                    post_image_path = os.path.join(images_dir, post_image_name)
                    pre_json_path  = os.path.join(labels_dir, base_id + "_pre_disaster.json")
                    pre_image_path = os.path.join(images_dir, pre_image_name)

                    # ensure all required files exist
                    if not os.path.isfile(post_json_path) or \
                       not os.path.isfile(post_image_path) or \
                       not os.path.isfile(pre_json_path) or \
                       not os.path.isfile(pre_image_path):
                        continue

                    # load JSON
                    with open(post_json_path, 'r') as f:
                        post_data = json.load(f)
                    # parse features
                    feats = post_data.get("features", {}).get(self.coord_key, [])
                    
                    for feat in feats:
                        damage_type = feat.get("properties", {}).get("subtype", "").lower()
                        if damage_type not in DAMAGE_LABELS:
                            # skip if "unclassified" or unknown
                            continue
                        label = DAMAGE_LABELS[damage_type]

                        # parse polygon
                        wkt_str = feat.get("wkt", None)
                        if wkt_str is None:
                            continue
                        polygon = wkt.loads(wkt_str)
                        # compute bounding box (minx, miny, maxx, maxy)
                        minx, miny, maxx, maxy = polygon.bounds

                        # store sample
                        samples.append({
                            "pre_img": pre_image_path,
                            "post_img": post_image_path,
                            "bbox": (minx, miny, maxx, maxy),
                            "label": label
                        })

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        pre_path = item["pre_img"]
        post_path = item["post_img"]
        (minx, miny, maxx, maxy) = item["bbox"]
        label = item["label"]

        # open images
        pre_img = Image.open(pre_path).convert("RGB")
        post_img = Image.open(post_path).convert("RGB")

        # center-based crop for each
        pre_crop = self._center_crop(pre_img, minx, miny, maxx, maxy, self.pre_crop_size)
        post_crop = self._center_crop(post_img, minx, miny, maxx, maxy, self.post_crop_size)

        # apply transforms
        pre_tensor = self.pre_transform(pre_crop)
        post_tensor = self.post_transform(post_crop)

        return pre_tensor, post_tensor, label

    def _center_crop(self, pil_img, minx, miny, maxx, maxy, crop_size):
        """
        Takes the bounding box, finds center, does a square crop of size (crop_size x crop_size).
        If it goes out of bounds, we clamp to the image edge.
        """
        width, height = pil_img.size  # (W, H)

        bb_width = maxx - minx
        bb_height = maxy - miny

        # bounding box center
        cx = minx + bb_width / 2.0
        cy = miny + bb_height / 2.0

        # We want a crop_size x crop_size region around (cx, cy)
        # top-left corner:
        half = crop_size / 2.0
        left = cx - half
        top = cy - half
        right = left + crop_size
        bottom = top + crop_size

        # clamp to image bounds
        left   = max(0, min(left, width - crop_size))
        top    = max(0, min(top, height - crop_size))
        right  = left + crop_size
        bottom = top + crop_size

        # Crop
        return pil_img.crop((left, top, right, bottom))

if __name__ == "__main__":
    dataset = XBDPatchDataset(root_dir="/home/pablos/Documents/uc3m/DammageAs/data/xBD")
    print(f"Total samples: {len(dataset)}")
    # Optionally, fetch one sample and print its shape
    pre_img, post_img, label = dataset[0]
    print(pre_img.shape, post_img.shape, label)