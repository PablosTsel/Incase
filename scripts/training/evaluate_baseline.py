#!/usr/bin/env python3
import os
import sys
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (confusion_matrix, classification_report,
                             f1_score, precision_score, recall_score,
                             cohen_kappa_score, balanced_accuracy_score,
                             accuracy_score, roc_auc_score, roc_curve, auc)
import time
from datetime import datetime
import json
from PIL import Image
import torchvision.transforms as T
from shapely import wkt
import glob

# Constants and definitions
DAMAGE_LABELS = {
    "no-damage": 0,
    "minor-damage": 1,
    "major-damage": 2,
    "destroyed": 3
}

# Dataset class
class XBDPatchDataset(Dataset):
    """
    On-the-fly dataset that:
      - Scans the dataset directory.
      - In hierarchical mode (flat_structure=False): expects subfolders per disaster (each with an 'images' and 'labels' folder).
      - In flat mode (flat_structure=True): expects root_dir to contain 'images' and 'labels' folders directly.
      - For each post-disaster JSON, enumerates building polygons, computes the bounding box, and center-crops the pre & post images.
      - Returns (pre_patch, post_patch, label) for each building.
    """
    def __init__(self,
                 root_dir,
                 pre_crop_size=128,
                 post_crop_size=224,
                 use_xy=True,
                 max_samples=None,
                 flat_structure=False,
                 augment=False):
        """
        :param root_dir: Directory where the data is stored.
           - If flat_structure is False: root_dir should contain subfolders for each disaster.
           - If True: root_dir should contain 'images' and 'labels' folders directly.
        :param pre_crop_size: Crop size for pre-disaster images.
        :param post_crop_size: Crop size for post-disaster images.
        :param use_xy: Use 'xy' coordinates if True; else use 'lng_lat'.
        :param max_samples: Optional limit on the number of samples.
        :param flat_structure: Whether the folder structure is flat.
        :param augment: If True, applies data augmentation (random horizontal flip and random rotation).
        """
        super().__init__()
        self.root_dir = root_dir
        self.pre_crop_size = pre_crop_size
        self.post_crop_size = post_crop_size
        self.coord_key = "xy" if use_xy else "lng_lat"
        self.max_samples = max_samples
        self.flat_structure = flat_structure
        self.augment = augment

        # Basic transforms - keeping it simple for compatibility
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

        self.samples = self._gather_samples()
        if self.max_samples is not None and len(self.samples) > self.max_samples:
            self.samples = random.sample(self.samples, self.max_samples)

    def _gather_samples(self):
        samples = []
        # First try to check if "test" directory has a flat structure
        images_dir = os.path.join(self.root_dir, "images")
        labels_dir = os.path.join(self.root_dir, "labels")
        
        if os.path.isdir(images_dir) and os.path.isdir(labels_dir):
            # This looks like a flat structure
            print(f"Detected flat structure at {self.root_dir}")
            label_files = [f for f in os.listdir(labels_dir) if f.endswith("_post_disaster.json")]
            
            for label_file in label_files:
                base_id = label_file.replace("_post_disaster.json", "")
                post_img_name = base_id + "_post_disaster.png"
                pre_img_name = base_id + "_pre_disaster.png"
                post_json_path = os.path.join(labels_dir, label_file)
                post_img_path = os.path.join(images_dir, post_img_name)
                pre_json_path = os.path.join(labels_dir, base_id + "_pre_disaster.json")
                pre_img_path = os.path.join(images_dir, pre_img_name)
                
                if not (os.path.isfile(post_json_path) and os.path.isfile(post_img_path)
                        and os.path.isfile(pre_json_path) and os.path.isfile(pre_img_path)):
                    continue
                    
                with open(post_json_path, 'r') as f:
                    post_data = json.load(f)
                    
                feats = post_data.get("features", {}).get(self.coord_key, [])
                for feat in feats:
                    damage_type = feat.get("properties", {}).get("subtype", "").lower()
                    if damage_type not in DAMAGE_LABELS:
                        continue
                        
                    label = DAMAGE_LABELS[damage_type]
                    wkt_str = feat.get("wkt", None)
                    if wkt_str is None:
                        continue
                        
                    polygon = wkt.loads(wkt_str)
                    minx, miny, maxx, maxy = polygon.bounds
                    samples.append({
                        "pre_img": pre_img_path,
                        "post_img": post_img_path,
                        "bbox": (minx, miny, maxx, maxy),
                        "label": label
                    })
        else:
            # Hierarchical structure: root_dir contains subfolders per disaster.
            try:
                disasters = [d for d in os.listdir(self.root_dir)
                            if os.path.isdir(os.path.join(self.root_dir, d))
                            and d.lower() != "spacenet_gt"]
                
                print(f"Found {len(disasters)} disaster folders")
                
                for disaster in disasters:
                    disaster_dir = os.path.join(self.root_dir, disaster)
                    images_dir = os.path.join(disaster_dir, "images")
                    labels_dir = os.path.join(disaster_dir, "labels")
                    
                    if not (os.path.isdir(images_dir) and os.path.isdir(labels_dir)):
                        print(f"Warning: Missing images or labels directory for disaster: {disaster}")
                        continue
                        
                    label_files = [f for f in os.listdir(labels_dir) if f.endswith("_post_disaster.json")]
                    print(f"Disaster {disaster}: Found {len(label_files)} label files")
                    
                    for label_file in label_files:
                        base_id = label_file.replace("_post_disaster.json", "")
                        post_img_name = base_id + "_post_disaster.png"
                        pre_img_name = base_id + "_pre_disaster.png"
                        post_json_path = os.path.join(labels_dir, label_file)
                        post_img_path = os.path.join(images_dir, post_img_name)
                        pre_json_path = os.path.join(labels_dir, base_id + "_pre_disaster.json")
                        pre_img_path = os.path.join(images_dir, pre_img_name)
                        
                        if not (os.path.isfile(post_json_path) and os.path.isfile(post_img_path)
                                and os.path.isfile(pre_json_path) and os.path.isfile(pre_img_path)):
                            continue
                            
                        with open(post_json_path, 'r') as f:
                            post_data = json.load(f)
                            
                        feats = post_data.get("features", {}).get(self.coord_key, [])
                        for feat in feats:
                            damage_type = feat.get("properties", {}).get("subtype", "").lower()
                            if damage_type not in DAMAGE_LABELS:
                                continue
                                
                            label = DAMAGE_LABELS[damage_type]
                            wkt_str = feat.get("wkt", None)
                            if wkt_str is None:
                                continue
                                
                            polygon = wkt.loads(wkt_str)
                            minx, miny, maxx, maxy = polygon.bounds
                            samples.append({
                                "pre_img": pre_img_path,
                                "post_img": post_img_path,
                                "bbox": (minx, miny, maxx, maxy),
                                "label": label,
                                "disaster": disaster  # Track which disaster this is from
                            })
            except Exception as e:
                print(f"Error gathering samples: {e}")
                
        print(f"Total gathered samples: {len(samples)}")
        # Check class distribution
        labels = [s["label"] for s in samples]
        unique_labels, counts = np.unique(labels, return_counts=True)
        print("Class distribution:")
        for label, count in zip(unique_labels, counts):
            class_name = [name for name, idx in DAMAGE_LABELS.items() if idx == label][0]
            print(f"  {class_name}: {count} samples ({count/len(samples)*100:.2f}%)")
            
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        pre_path = item["pre_img"]
        post_path = item["post_img"]
        (minx, miny, maxx, maxy) = item["bbox"]
        label = item["label"]

        try:
            pre_img = Image.open(pre_path).convert("RGB")
            post_img = Image.open(post_path).convert("RGB")

            pre_crop = self._center_crop(pre_img, minx, miny, maxx, maxy, self.pre_crop_size)
            post_crop = self._center_crop(post_img, minx, miny, maxx, maxy, self.post_crop_size)

            # Use consistent random state for both images
            if self.augment:
                seed = np.random.randint(2147483647)
                random.seed(seed)
                torch.manual_seed(seed)
                
            pre_tensor = self.pre_transform(pre_crop)
            
            if self.augment:
                # Reset the seed for the second transform to ensure same transformation
                random.seed(seed)
                torch.manual_seed(seed)
                
            post_tensor = self.post_transform(post_crop)

            return pre_tensor, post_tensor, label
            
        except Exception as e:
            print(f"Error processing item {idx}: {e}")
            # Return a placeholder in case of error
            placeholder_pre = torch.zeros(3, self.pre_crop_size, self.pre_crop_size)
            placeholder_post = torch.zeros(3, self.post_crop_size, self.post_crop_size)
            return placeholder_pre, placeholder_post, label

    def _center_crop(self, pil_img, minx, miny, maxx, maxy, crop_size):
        width, height = pil_img.size
        bb_width = maxx - minx
        bb_height = maxy - miny
        cx = minx + bb_width / 2.0
        cy = miny + bb_height / 2.0
        
        half = crop_size / 2.0
        left = max(0, min(cx - half, width - crop_size))
        top = max(0, min(cy - half, height - crop_size))
        right = left + crop_size
        bottom = top + crop_size
        return pil_img.crop((left, top, right, bottom))

# Model definition
class AttentionFusion(nn.Module):
    def __init__(self, in_channels):
        super(AttentionFusion, self).__init__()
        self.attention = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
    def forward(self, pre_feat, post_feat):
        # Make sure the spatial dimensions match before concatenating
        if pre_feat.shape[2:] != post_feat.shape[2:]:
            # Resize post_feat to match pre_feat's spatial dimensions
            post_feat = F.interpolate(post_feat, size=pre_feat.shape[2:], 
                                      mode='bilinear', align_corners=False)
            
        # Concatenate features along channel dimension
        concat_feat = torch.cat([pre_feat, post_feat], dim=1)
        # Generate attention weights
        attn_weights = self.attention(concat_feat)
        # Apply attention to post features
        weighted_post = post_feat * attn_weights
        # Combine pre and weighted post features
        fused_feat = pre_feat + weighted_post
        return fused_feat

class BaselineModel(nn.Module):
    def __init__(self, num_classes=4, pretrained=True, dropout_rate=0.5):
        super(BaselineModel, self).__init__()
        
        # Use a more powerful backbone (ResNet50)
        try:
            # For torch 1.13+
            import torchvision.models as models
            base_model = models.resnet50(weights='IMAGENET1K_V2' if pretrained else None)
        except TypeError:
            # For older torch versions
            import torchvision.models as models
            base_model = models.resnet50(pretrained=pretrained)
        
        # Pre-disaster branch
        self.pre_branch = nn.Sequential(*list(base_model.children())[:-2])
        
        # Post-disaster branch (same architecture but separate weights)
        try:
            # For torch 1.13+
            import torchvision.models as models
            post_model = models.resnet50(weights='IMAGENET1K_V2' if pretrained else None)
        except TypeError:
            # For older torch versions
            import torchvision.models as models
            post_model = models.resnet50(pretrained=pretrained)
        
        self.post_branch = nn.Sequential(*list(post_model.children())[:-2])
        
        # Get the number of features from the backbone
        self.feature_dim = base_model.fc.in_features  # 2048 for ResNet50
        
        # Attention fusion module to combine features from both branches
        self.attention_fusion = AttentionFusion(self.feature_dim)
        
        # Global average pooling to reduce spatial dimensions
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # MLP head with dropout for better generalization
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )
        
        # Initialize weights of the classifier
        self._initialize_weights(self.classifier)
        
    def _initialize_weights(self, module):
        for m in module.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
    def forward(self, pre_images, post_images):
        # Extract features from pre-disaster images
        pre_features = self.pre_branch(pre_images)
        
        # Extract features from post-disaster images
        post_features = self.post_branch(post_images)
        
        # Fuse features with attention mechanism
        fused_features = self.attention_fusion(pre_features, post_features)
        
        # Global average pooling
        pooled_features = self.gap(fused_features).view(-1, self.feature_dim)
        
        # Classification
        output = self.classifier(pooled_features)
        
        return output

# Evaluation functions
def seed_everything(seed=42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"Random seed set to {seed}")

def test_time_augmentation(model, pre_img, post_img, device, num_augments=5):
    """Apply test-time augmentation by running multiple forward passes with different augmentations."""
    model.eval()
    batch_size = pre_img.size(0)
    
    # Initialize with original predictions
    with torch.no_grad():
        outputs_original = model(pre_img, post_img)
    
    all_outputs = [outputs_original]
    
    # Horizontal flip augmentation
    with torch.no_grad():
        pre_hflip = torch.flip(pre_img, dims=[3])
        post_hflip = torch.flip(post_img, dims=[3])
        outputs_hflip = model(pre_hflip, post_hflip)
        all_outputs.append(outputs_hflip)
    
    # Vertical flip augmentation
    with torch.no_grad():
        pre_vflip = torch.flip(pre_img, dims=[2])
        post_vflip = torch.flip(post_img, dims=[2])
        outputs_vflip = model(pre_vflip, post_vflip)
        all_outputs.append(outputs_vflip)
    
    # Both horizontal and vertical flip
    with torch.no_grad():
        pre_hvflip = torch.flip(pre_img, dims=[2, 3])
        post_hvflip = torch.flip(post_img, dims=[2, 3])
        outputs_hvflip = model(pre_hvflip, post_hvflip)
        all_outputs.append(outputs_hvflip)
    
    # Center crop augmentation (90% of image)
    if num_augments > 4:
        with torch.no_grad():
            h, w = pre_img.shape[2:]
            ch, cw = int(h * 0.1), int(w * 0.1)
            pre_crop = pre_img[:, :, ch:h-ch, cw:w-cw]
            post_crop = post_img[:, :, ch:h-ch, cw:w-cw]
            pre_crop = F.interpolate(pre_crop, size=(h, w), mode='bilinear', align_corners=False)
            post_crop = F.interpolate(post_crop, size=(h, w), mode='bilinear', align_corners=False)
            outputs_crop = model(pre_crop, post_crop)
            all_outputs.append(outputs_crop)
    
    # Average the predictions
    outputs = torch.stack(all_outputs).mean(dim=0)
    return outputs

def evaluate_model(model, dataloader, device, use_tta=False):
    """Evaluate model with optional test-time augmentation."""
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    all_probs = []  # For ROC curves
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Evaluating", leave=True)
        for pre_imgs, post_imgs, labels in pbar:
            pre_imgs = pre_imgs.to(device)
            post_imgs = post_imgs.to(device)
            labels = labels.to(device)
            
            # Either use test-time augmentation or standard forward pass
            if use_tta:
                outputs = test_time_augmentation(model, pre_imgs, post_imgs, device)
            else:
                outputs = model(pre_imgs, post_imgs)
            
            # Get predictions and probabilities
            probabilities = F.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probabilities.cpu().numpy())
            
            # Update progress bar
            pbar.set_postfix({"Acc": f"{100 * correct / total:.2f}%"})
    
    accuracy = 100 * correct / total if total > 0 else 0
    return accuracy, np.array(all_preds), np.array(all_labels), np.array(all_probs)

def plot_confusion_matrix(cm, target_names, save_path, normalize=False, title=None):
    """
    Generate and save a confusion matrix plot.
    
    Parameters:
    - cm: Confusion matrix from sklearn
    - target_names: List of class names
    - save_path: Path to save the plot
    - normalize: Boolean, whether to normalize values
    - title: Title for the plot
    """
    if normalize:
        cm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-6)
        fmt = '.2f'
    else:
        fmt = 'd'
    
    plt.figure(figsize=(10, 8))
    sns.set(font_scale=1.2)
    sns.heatmap(cm, annot=True, fmt=fmt, cmap="Blues",
                xticklabels=target_names, yticklabels=target_names)
    
    plt.ylabel('True Label', fontsize=14)
    plt.xlabel('Predicted Label', fontsize=14)
    
    if title:
        plt.title(title, fontsize=16)
    else:
        plt.title("Confusion Matrix", fontsize=16)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_per_class_metrics(metrics, metric_name, target_names, save_path):
    """Generate and save per-class metrics bar chart."""
    plt.figure(figsize=(12, 6))
    x = np.arange(len(target_names))
    
    bar_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    bars = plt.bar(x, metrics, color=bar_colors, width=0.6)
    
    # Add value labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', fontsize=10)
    
    plt.axhline(y=np.mean(metrics), color='r', linestyle='--', label=f'Mean: {np.mean(metrics):.3f}')
    
    plt.xticks(x, target_names, rotation=30)
    plt.ylim(0, min(1, max(metrics) + 0.15))
    
    plt.xlabel("Classes", fontsize=12)
    plt.ylabel(metric_name, fontsize=12)
    plt.title(f"Per-Class {metric_name}", fontsize=14)
    
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_roc_curves(all_probs, all_labels, target_names, save_path):
    """Generate and save ROC curves for each class."""
    plt.figure(figsize=(12, 8))
    
    # One-hot encode the labels for multi-class ROC
    n_classes = len(target_names)
    y_true_onehot = np.eye(n_classes)[all_labels]
    
    # Colors for each class
    colors = ['blue', 'orange', 'green', 'red']
    
    # Calculate ROC curve and ROC area for each class
    for i, color, target_name in zip(range(n_classes), colors, target_names):
        fpr, tpr, _ = roc_curve(y_true_onehot[:, i], all_probs[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=color, lw=2,
                 label=f'{target_name} (AUC = {roc_auc:.3f})')
    
    # Plot the diagonal
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Receiver Operating Characteristic (ROC) Curves', fontsize=14)
    plt.legend(loc="lower right")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def analyze_errors(all_preds, all_labels, all_probs, target_names, save_path):
    """Analyze prediction errors to find common patterns."""
    # Create a figure for error analysis
    plt.figure(figsize=(12, 8))
    
    # Get indices of errors
    error_idx = np.where(all_preds != all_labels)[0]
    correct_idx = np.where(all_preds == all_labels)[0]
    
    # Calculate error rates for each true class
    error_rates = []
    error_types = []
    
    for true_class in range(len(target_names)):
        # Get all samples of this true class
        class_samples = np.where(all_labels == true_class)[0]
        # Find errors among these samples
        class_errors = np.intersect1d(class_samples, error_idx)
        # Calculate error rate
        error_rate = len(class_errors) / len(class_samples) if len(class_samples) > 0 else 0
        error_rates.append(error_rate)
        
        # Find common misclassifications
        if len(class_errors) > 0:
            error_preds = all_preds[class_errors]
            unique_preds, counts = np.unique(error_preds, return_counts=True)
            most_common_idx = np.argmax(counts)
            most_common_class = unique_preds[most_common_idx]
            error_types.append(f"{target_names[true_class]} → {target_names[most_common_class]}")
        else:
            error_types.append("No errors")
    
    # Plot error rates
    plt.subplot(2, 1, 1)
    bars = plt.bar(target_names, error_rates, color=['#ff9999', '#66b3ff', '#99ff99', '#ffcc99'])
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{height:.2f}', ha='center', fontsize=10)
    
    plt.title("Error Rate by True Class", fontsize=14)
    plt.ylabel("Error Rate", fontsize=12)
    plt.ylim(0, max(error_rates) + 0.1)
    
    # Analyze confidence of correct vs. incorrect predictions
    plt.subplot(2, 1, 2)
    
    # Get confidence (probability of predicted class)
    confidences = np.array([all_probs[i, pred] for i, pred in enumerate(all_preds)])
    
    # Split by correct/incorrect
    correct_conf = confidences[correct_idx]
    error_conf = confidences[error_idx]
    
    # Create histogram
    plt.hist(correct_conf, bins=20, alpha=0.5, label=f'Correct (n={len(correct_conf)})', color='green')
    plt.hist(error_conf, bins=20, alpha=0.5, label=f'Incorrect (n={len(error_conf)})', color='red')
    
    plt.xlabel("Prediction Confidence", fontsize=12)
    plt.ylabel("Count", fontsize=12)
    plt.title("Confidence Distribution for Correct vs. Incorrect Predictions", fontsize=14)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    
    # Print error analysis summary
    print("\nError Analysis:")
    print("  Most common misclassification patterns:")
    for i, error in enumerate(error_types):
        print(f"  - {error} (Error rate: {error_rates[i]:.2f})")

def create_versioned_directory(base_path, prefix="evaluationTry"):
    """Create a versioned directory with incremented try number if base exists."""
    i = 1
    while True:
        dir_name = f"{prefix}{i}"
        full_path = os.path.join(base_path, dir_name)
        if not os.path.exists(full_path):
            os.makedirs(full_path, exist_ok=True)
            return full_path, i
        i += 1

def find_best_model(models_dir):
    """Find the latest best model in the models directory."""
    # First check for baseline_best.pt
    baseline_path = os.path.join(models_dir, "baseline_best.pt")
    if os.path.exists(baseline_path):
        return baseline_path
    
    # Otherwise, look for the best_model_epoch_X.pt file with the highest epoch number
    model_files = glob.glob(os.path.join(models_dir, "best_model_epoch_*.pt"))
    if not model_files:
        return None
    
    # Extract epoch numbers and find the highest
    epoch_numbers = []
    for file_path in model_files:
        filename = os.path.basename(file_path)
        try:
            epoch = int(filename.replace("best_model_epoch_", "").replace(".pt", ""))
            epoch_numbers.append((epoch, file_path))
        except ValueError:
            continue
    
    if not epoch_numbers:
        return None
    
    # Return the path with the highest epoch number
    return sorted(epoch_numbers, key=lambda x: x[0], reverse=True)[0][1]

def main():
    # Set random seed for reproducibility
    seed_everything(42)
    
    # Get project root directory
    project_root = os.path.abspath(os.path.dirname(__file__))
    # Go up two levels from scripts/training to the project root
    project_root = os.path.abspath(os.path.join(project_root, "..", ".."))
    
    # Configuration
    use_tta = True  # Use test-time augmentation
    
    # Find the latest training directory and best model within it
    output_dir = os.path.join(project_root, "output")
    training_dirs = sorted([d for d in os.listdir(output_dir) 
                           if os.path.isdir(os.path.join(output_dir, d)) and d.startswith("trainingTry")],
                          key=lambda x: int(x.replace("trainingTry", "")))
    
    if not training_dirs:
        print("No training directories found. Please train a model first.")
        return
    
    latest_training_dir = os.path.join(output_dir, training_dirs[-1])
    models_dir = os.path.join(latest_training_dir, "models")
    
    # Find the best model in the latest training directory
    best_model_path = find_best_model(models_dir)
    
    if not best_model_path:
        print(f"No model checkpoint found in {models_dir}")
        return
    
    print(f"Using model: {best_model_path}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Create timestamp for saving results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create versioned results directory
    base_results_dir = os.path.join(project_root, "evaluation_results")
    os.makedirs(base_results_dir, exist_ok=True)
    results_dir, try_num = create_versioned_directory(base_results_dir)
    
    # Test data is stored in a hierarchical structure under data/test
    test_data_dir = os.path.join(project_root, "data", "test")
    print(f"Loading evaluation data from: {test_data_dir}")

    # Create test dataset without augmentation
    test_dataset = XBDPatchDataset(
        root_dir=test_data_dir,
        pre_crop_size=128,
        post_crop_size=224,
        use_xy=True,
        max_samples=None,  # Use all samples for evaluation
        flat_structure=False,  # Test data is hierarchical
        augment=False        # No augmentation during evaluation
    )
    print(f"Total evaluation samples: {len(test_dataset)}")

    # Create test dataloader with optimizations
    test_loader = DataLoader(
        test_dataset, 
        batch_size=32,  # Increased batch size for faster evaluation
        shuffle=False, 
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )

    # Initialize model
    model = BaselineModel(num_classes=4).to(device)
    
    # Load checkpoint
    if os.path.isfile(best_model_path):
        checkpoint = torch.load(best_model_path, map_location=device)
        model.load_state_dict(checkpoint, strict=False)
        print(f"Loaded model weights from {best_model_path}")
    else:
        print(f"Model checkpoint not found at {best_model_path}")
        return

    # Evaluate model
    print(f"Starting evaluation{' with test-time augmentation' if use_tta else ''}...")
    start_time = time.time()
    accuracy, preds, labels, probs = evaluate_model(model, test_loader, device, use_tta=use_tta)
    eval_time = time.time() - start_time
    
    print(f"\nEvaluation completed in {eval_time:.2f} seconds")
    print(f"Final Evaluation Accuracy: {accuracy:.2f}%\n")

    # Compute detailed metrics
    target_names = list(DAMAGE_LABELS.keys())
    
    # Check if there are any predictions
    if len(labels) == 0:
        print("No samples were evaluated. Exiting...")
        return

    # Create confusion matrix
    conf_matrix = confusion_matrix(labels, preds)
    print("Confusion Matrix:")
    print(conf_matrix)

    # Create classification report with proper handling of empty arrays
    if len(labels) > 0 and len(np.unique(labels)) > 0:
        report = classification_report(labels, preds, target_names=target_names, digits=4)
        print("\nClassification Report:")
        print(report)

        # Calculate additional metrics
        macro_f1 = f1_score(labels, preds, average='macro')
        weighted_f1 = f1_score(labels, preds, average='weighted')
        macro_precision = precision_score(labels, preds, average='macro')
        weighted_precision = precision_score(labels, preds, average='weighted')
        macro_recall = recall_score(labels, preds, average='macro')
        weighted_recall = recall_score(labels, preds, average='weighted')
        kappa = cohen_kappa_score(labels, preds)
        bal_accuracy = balanced_accuracy_score(labels, preds)
    else:
        print("No valid labels or predictions to calculate metrics.")
        macro_f1 = weighted_f1 = macro_precision = weighted_precision = macro_recall = weighted_recall = kappa = bal_accuracy = 0.0

    print("Additional Metrics:")
    print(f"Macro F1 Score: {macro_f1:.4f}")
    print(f"Weighted F1 Score: {weighted_f1:.4f}")
    print(f"Macro Precision: {macro_precision:.4f}")
    print(f"Weighted Precision: {weighted_precision:.4f}")
    print(f"Macro Recall: {macro_recall:.4f}")
    print(f"Weighted Recall: {weighted_recall:.4f}")
    print(f"Cohen's Kappa: {kappa:.4f}")
    print(f"Balanced Accuracy: {bal_accuracy:.4f}")

    # Save detailed results to a text file
    results_path = os.path.join(results_dir, f"evaluation_results_try{try_num}.txt")
    with open(results_path, 'w') as f:
        f.write(f"Evaluation Results (Try {try_num})\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model: {best_model_path}\n")
        f.write(f"Test-Time Augmentation: {use_tta}\n\n")
        
        if len(labels) > 0:
            f.write("Confusion Matrix:\n")
            f.write(str(conf_matrix) + "\n\n")
            
            if len(np.unique(labels)) > 0:
                f.write("Classification Report:\n")
                f.write(report + "\n\n")
        else:
            f.write("No samples were evaluated.\n\n")
        
        f.write("Additional Metrics:\n")
        f.write(f"Accuracy: {accuracy:.4f}%\n")
        f.write(f"Macro F1 Score: {macro_f1:.4f}\n")
        f.write(f"Weighted F1 Score: {weighted_f1:.4f}\n")
        f.write(f"Macro Precision: {macro_precision:.4f}\n")
        f.write(f"Weighted Precision: {weighted_precision:.4f}\n")
        f.write(f"Macro Recall: {macro_recall:.4f}\n")
        f.write(f"Weighted Recall: {weighted_recall:.4f}\n")
        f.write(f"Cohen's Kappa: {kappa:.4f}\n")
        f.write(f"Balanced Accuracy: {bal_accuracy:.4f}\n")
    
    print(f"Detailed evaluation results saved to {results_path}")

    # Create visualizations 
    cm_save_path = os.path.join(results_dir, "confusion_matrix.png")
    plot_confusion_matrix(conf_matrix, target_names, cm_save_path, 
                         title=f"Confusion Matrix (Accuracy: {accuracy:.2f}%)")
    
    cm_norm_save_path = os.path.join(results_dir, "confusion_matrix_normalized.png")
    plot_confusion_matrix(conf_matrix, target_names, cm_norm_save_path, normalize=True,
                         title="Normalized Confusion Matrix")

    # Plot per-class metrics
    try:
        per_class_f1 = f1_score(labels, preds, average=None, labels=range(len(target_names)))
        per_class_precision = precision_score(labels, preds, average=None, labels=range(len(target_names)))
        per_class_recall = recall_score(labels, preds, average=None, labels=range(len(target_names)))
        
        f1_save_path = os.path.join(results_dir, "per_class_f1.png")
        plot_per_class_metrics(per_class_f1, "F1 Score", target_names, f1_save_path)
        
        precision_save_path = os.path.join(results_dir, "per_class_precision.png")
        plot_per_class_metrics(per_class_precision, "Precision", target_names, precision_save_path)
        
        recall_save_path = os.path.join(results_dir, "per_class_recall.png")
        plot_per_class_metrics(per_class_recall, "Recall", target_names, recall_save_path)
        
        # Plot ROC curves for multi-class classification
        roc_save_path = os.path.join(results_dir, "roc_curves.png")
        plot_roc_curves(probs, labels, target_names, roc_save_path)
        
        # Analyze errors
        error_save_path = os.path.join(results_dir, "error_analysis.png")
        analyze_errors(preds, labels, probs, target_names, error_save_path)
    except Exception as e:
        print(f"Error generating metric plots: {e}")
    
    # Randomly sample predictions to show variety if we have predictions
    if len(preds) > 0:
        num_samples = min(15, len(preds))
        sample_indices = random.sample(range(len(preds)), num_samples)
        
        print("\nSample Predictions (randomly selected):")
        print(f"{'Index':<8} {'True Label':<15} {'Predicted':<15} {'Confidence':<10} {'Correct':<8}")
        print("-" * 60)
        
        for i in sample_indices:
            pred_class = target_names[preds[i]]
            true_class = target_names[labels[i]]
            confidence = probs[i, preds[i]]
            correct = "✓" if preds[i] == labels[i] else "✗"
            print(f"{i:<8} {true_class:<15} {pred_class:<15} {confidence:.4f}      {correct}")

if __name__ == "__main__":
    main()