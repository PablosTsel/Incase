#!/usr/bin/env python3
# Building Localization Model using U-Net architecture
# For xBD satellite imagery

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import numpy as np
import random
from datetime import datetime
import time
import json
from PIL import Image
import torchvision.transforms as T
from shapely import wkt
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import jaccard_score


# Set random seeds for reproducibility
def seed_everything(seed=42):
    """
    Set random seeds for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"Random seed set to {seed}")


# U-Net architecture components
class DoubleConv(nn.Module):
    """
    Double convolution block for U-Net.
    (Conv2d -> BatchNorm -> ReLU) × 2
    """
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """
    Downsampling block for U-Net.
    MaxPool -> DoubleConv
    """
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """
    Upsampling block for U-Net.
    Upsample -> Concatenate with skip connection -> DoubleConv
    """
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()
        
        # Use either transposed convolution or bilinear upsampling
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)
            
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # Adjust the dimensions if needed (account for odd sizes)
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]
        
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])
        
        # Concatenate along the channel dimension
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """
    Output convolution block.
    Simple 1x1 convolution to get desired number of output channels.
    """
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


# Complete U-Net Model
class UNet(nn.Module):
    """
    U-Net architecture for semantic segmentation.
    
    This implementation follows the original U-Net paper architecture
    with optional modifications for performance.
    """
    def __init__(self, n_channels=3, n_classes=1, bilinear=True):
        """
        Initialize U-Net model.
        
        Args:
            n_channels (int): Number of input channels (3 for RGB images)
            n_classes (int): Number of output classes (1 for binary segmentation)
            bilinear (bool): Whether to use bilinear upsampling or transposed convolutions
        """
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        
        # Initial double convolution
        self.inc = DoubleConv(n_channels, 64)
        
        # Encoder (downsampling path)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512 if bilinear else 1024)
        
        # Decoder (upsampling path)
        self.up1 = Up(1024, 256, bilinear)
        self.up2 = Up(512, 128, bilinear)
        self.up3 = Up(256, 64, bilinear)
        self.up4 = Up(128, 64, bilinear)
        
        # Final 1x1 convolution to get desired number of output channels
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        # Encoder path
        x1 = self.inc(x)       # Initial features
        x2 = self.down1(x1)    # First downsampling
        x3 = self.down2(x2)    # Second downsampling
        x4 = self.down3(x3)    # Third downsampling
        x5 = self.down4(x4)    # Fourth downsampling (bottleneck)
        
        # Decoder path with skip connections
        x = self.up1(x5, x4)   # First upsampling + skip from x4
        x = self.up2(x, x3)    # Second upsampling + skip from x3
        x = self.up3(x, x2)    # Third upsampling + skip from x2
        x = self.up4(x, x1)    # Fourth upsampling + skip from x1
        
        # Final convolution
        logits = self.outc(x)
        
        return logits


# Dataset for building segmentation
class XBDSegmentationDataset(Dataset):
    """
    Dataset for building segmentation from xBD satellite imagery.
    
    This dataset loads pre-disaster images and creates binary masks
    for buildings based on the polygon data in the JSON files.
    """
    def __init__(self, root_dir, image_size=512, transform=None):
        """
        Initialize the dataset.
        
        Args:
            root_dir (str): Directory containing the dataset
            image_size (int): Size to resize the images to
            transform (callable, optional): Optional transform to be applied on the images
        """
        self.root_dir = root_dir
        self.image_size = image_size
        self.transform = transform
        self.samples = self._gather_samples()
        
    def _gather_samples(self):
        """
        Scan the dataset directory and gather valid samples.
        
        Returns:
            list: List of dictionaries with paths to images and corresponding label JSONs
        """
        samples = []
        
        # Check if we have hierarchical structure
        disasters = [d for d in os.listdir(self.root_dir)
                    if os.path.isdir(os.path.join(self.root_dir, d))
                    and d.lower() != "spacenet_gt"]
        
        if not disasters:
            # Try flat structure
            images_dir = os.path.join(self.root_dir, "images")
            labels_dir = os.path.join(self.root_dir, "labels")
            
            if os.path.isdir(images_dir) and os.path.isdir(labels_dir):
                # Process files in flat structure
                pre_images = [f for f in os.listdir(images_dir) if f.endswith("_pre_disaster.png")]
                
                for img_name in pre_images:
                    base_id = img_name.replace("_pre_disaster.png", "")
                    pre_img_path = os.path.join(images_dir, img_name)
                    pre_json_path = os.path.join(labels_dir, base_id + "_pre_disaster.json")
                    
                    if os.path.isfile(pre_img_path) and os.path.isfile(pre_json_path):
                        samples.append({
                            "image": pre_img_path,
                            "json": pre_json_path
                        })
        else:
            # Process hierarchical structure
            for disaster in disasters:
                disaster_dir = os.path.join(self.root_dir, disaster)
                images_dir = os.path.join(disaster_dir, "images")
                labels_dir = os.path.join(disaster_dir, "labels")
                
                if not (os.path.isdir(images_dir) and os.path.isdir(labels_dir)):
                    continue
                
                pre_images = [f for f in os.listdir(images_dir) if f.endswith("_pre_disaster.png")]
                
                for img_name in pre_images:
                    base_id = img_name.replace("_pre_disaster.png", "")
                    pre_img_path = os.path.join(images_dir, img_name)
                    pre_json_path = os.path.join(labels_dir, base_id + "_pre_disaster.json")
                    
                    if os.path.isfile(pre_img_path) and os.path.isfile(pre_json_path):
                        samples.append({
                            "image": pre_img_path,
                            "json": pre_json_path,
                            "disaster": disaster
                        })
        
        print(f"Found {len(samples)} samples for building segmentation")
        return samples
    
    def _create_mask_from_polygons(self, json_path, img_size):
        """
        Create a binary mask from building polygons in a JSON file.
        
        Args:
            json_path (str): Path to the JSON file with building polygons
            img_size (tuple): Size of the image (width, height)
            
        Returns:
            np.ndarray: Binary mask where 1 indicates building pixels
        """
        from rasterio.features import rasterize
        import rasterio.transform
        
        width, height = img_size
        mask = np.zeros((height, width), dtype=np.uint8)
        
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            # Extract building polygons
            features = data.get("features", {}).get("xy", [])
            
            # Create list of polygons for rasterization
            polygons = []
            for feature in features:
                # Check if it's a building
                properties = feature.get("properties", {})
                if properties.get("feature_type") == "building":
                    wkt_str = feature.get("wkt")
                    if wkt_str:
                        polygon = wkt.loads(wkt_str)
                        polygons.append(polygon)
            
            
            # Rasterize all building polygons
            if polygons:
                transform = rasterio.transform.from_bounds(0, 0, width, height, width, height)
                mask = rasterize(
                    shapes=[(polygon, 1) for polygon in polygons],
                    out_shape=(height, width),
                    transform=transform,
                    fill=0,
                    all_touched=False,
                    dtype=np.uint8
                )
        except Exception as e:
            print(f"Error creating mask from {json_path}: {e}")
        
        return mask
        
    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.samples)
    
    def __getitem__(self, idx):
        """
        Get a sample from the dataset.
        
        Returns:
            tuple: (image, mask) where mask is the ground truth segmentation
        """
        sample = self.samples[idx]
        
        # Load image
        image = Image.open(sample["image"]).convert("RGB")
        width, height = image.size
        
        # Create binary mask for buildings
        mask = self._create_mask_from_polygons(sample["json"], (width, height))
        mask = Image.fromarray(mask)
        
        # Resize both image and mask
        image = image.resize((self.image_size, self.image_size), Image.BICUBIC)
        mask = mask.resize((self.image_size, self.image_size), Image.NEAREST)
        
        # Convert to tensors
        image = T.ToTensor()(image)
        mask = torch.from_numpy(np.array(mask)).float().unsqueeze(0)  # Add channel dimension
        
        # Apply additional transforms if specified
        if self.transform:
            image = self.transform(image)
        
        return image, mask


# Training functions
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device, output_dir):
    """
    Train the U-Net model.
    
    Args:
        model: The U-Net model
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        criterion: Loss function
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        num_epochs: Number of epochs to train for
        device: Device to train on (cuda or cpu)
        output_dir: Directory to save models
        
    Returns:
        tuple: (training history, path to saved model)
    """
    # Make sure output_dir is valid and writable
    try:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        # Test if directory is writable
        test_file = os.path.join(output_dir, '.test_write')
        with open(test_file, 'w') as f:
            f.write('test')
        os.remove(test_file)
    except Exception as e:
        print(f"Warning: Output directory {output_dir} is not writable: {e}")
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        output_dir = project_root
        print(f"Using project root directory instead: {output_dir}")
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_iou': []
    }
    
    best_val_iou = 0.0
    best_model_path = None
    
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        
        # Training phase
        model.train()
        train_loss = 0.0
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [TRAIN]")
        
        for inputs, masks in train_pbar:
            inputs = inputs.to(device)
            masks = masks.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, masks)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            train_pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        train_loss /= len(train_loader.dataset)
        history['train_loss'].append(train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_iou = 0.0
        
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [VAL]")
        
        with torch.no_grad():
            for inputs, masks in val_pbar:
                inputs = inputs.to(device)
                masks = masks.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, masks)
                
                val_loss += loss.item() * inputs.size(0)
                
                # Calculate IoU
                preds = (torch.sigmoid(outputs) > 0.5).float()
                iou = calculate_iou(preds, masks)
                val_iou += iou * inputs.size(0)
                
                val_pbar.set_postfix({"loss": f"{loss.item():.4f}", "iou": f"{iou:.4f}"})
        
        val_loss /= len(val_loader.dataset)
        val_iou /= len(val_loader.dataset)
        
        history['val_loss'].append(val_loss)
        history['val_iou'].append(val_iou)
        
        # Update learning rate
        scheduler.step(val_iou)
        
        print(f"Epoch {epoch+1}/{num_epochs} - "
              f"Train Loss: {train_loss:.4f}, "
              f"Val Loss: {val_loss:.4f}, "
              f"Val IoU: {val_iou:.4f}")
        
        # Save best model
        if val_iou > best_val_iou:
            best_val_iou = val_iou
            best_model_path = os.path.join(output_dir, "best_unet_model.pt")
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved with IoU: {best_val_iou:.4f}")
    
    # Save final model if no best model was saved
    if best_model_path is None:
        best_model_path = os.path.join(output_dir, "final_unet_model.pt")
        torch.save(model.state_dict(), best_model_path)
        print(f"No improvement in IoU, saving final model instead")
    
    return history, best_model_path


def calculate_iou(outputs, targets):
    """
    Calculate Intersection over Union (IoU) metric.
    
    Args:
        outputs (torch.Tensor): Model predictions (after sigmoid and thresholding)
        targets (torch.Tensor): Ground truth masks
        
    Returns:
        float: IoU score
    """
    # Flatten tensors
    outputs = outputs.cpu().view(-1).numpy()
    targets = targets.cpu().view(-1).numpy()
    
    # Calculate IoU
    intersection = np.logical_and(outputs, targets).sum()
    union = np.logical_or(outputs, targets).sum()
    
    # Avoid division by zero
    if union == 0:
        return 0.0
    
    iou = intersection / union
    return iou


def visualize_predictions(model, dataset, num_samples=5, device='cuda', output_dir=None, project_root=None):
    """
    Visualize model predictions on sample images.
    
    Args:
        model: Trained model
        dataset: Dataset to sample from
        num_samples: Number of samples to visualize
        device: Device to run inference on
        output_dir: Directory to save visualization
        project_root: Project root directory for additional backup save
    """
    if output_dir is None:
        output_dir = '.'  # Default to current directory if not specified
    
    # Make sure output_dir exists and is writable
    try:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        # Test if directory is writable
        test_file = os.path.join(output_dir, '.test_write')
        with open(test_file, 'w') as f:
            f.write('test')
        os.remove(test_file)
    except Exception as e:
        print(f"Warning: Output directory {output_dir} is not writable: {e}")
        print("Using current directory instead.")
        output_dir = '.'
        
    model.eval()
    indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))
    
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5*num_samples))
    
    with torch.no_grad():
        for i, idx in enumerate(indices):
            image, mask = dataset[idx]
            image = image.unsqueeze(0).to(device)
            
            # Get prediction
            output = model(image)
            prediction = torch.sigmoid(output) > 0.5
            
            # Convert tensors to numpy for visualization
            image = image.squeeze().cpu().permute(1, 2, 0).numpy()
            mask = mask.squeeze().cpu().numpy()
            prediction = prediction.squeeze().cpu().numpy()
            
            # Normalize image
            image = (image - image.min()) / (image.max() - image.min())
            
            # Plot
            axes[i, 0].imshow(image)
            axes[i, 0].set_title('Input Image')
            axes[i, 0].axis('off')
            
            axes[i, 1].imshow(mask, cmap='gray')
            axes[i, 1].set_title('Ground Truth')
            axes[i, 1].axis('off')
            
            axes[i, 2].imshow(prediction, cmap='gray')
            axes[i, 2].set_title('Prediction')
            axes[i, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'prediction_visualization.png'))
    
    # Also save to project root if provided
    if project_root is not None:
        try:
            plt.savefig(os.path.join(project_root, 'unet_predictions.png'))
            print(f"Prediction visualization also saved to {os.path.join(project_root, 'unet_predictions.png')}")
        except Exception as e:
            print(f"Could not save prediction visualization to project root: {e}")
    
    plt.close()


def main():
    """
    Main function to train and evaluate the U-Net model for building localization.
    """
    # Set random seed for reproducibility
    seed_everything(42)
    
    # Get project root directory
    project_root = os.path.abspath(os.path.dirname(__file__))
    # Go up two levels from current script to the project root
    project_root = os.path.abspath(os.path.join(project_root, "..", ".."))
    
    # Dataset directory
    data_dir = os.path.join(project_root, "data", "xBD")
    
    # Configuration
    image_size = 512
    batch_size = 4
    num_epochs = 35  # Reduced from 20 to 10 for faster training
    learning_rate = 1e-4
    val_ratio = 0.2
    
    # Create output directory
    output_dir = os.path.join(project_root, "output", "localization")
    try:
        os.makedirs(output_dir, exist_ok=True)
    except Exception as e:
        print(f"Warning: Could not create output directory {output_dir}: {e}")
        print(f"Using project root directory instead.")
        output_dir = project_root
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create dataset and dataloaders
    dataset = XBDSegmentationDataset(root_dir=data_dir, image_size=image_size)
    
    # Split into train and validation sets
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    random.shuffle(indices)
    split = int(val_ratio * dataset_size)
    train_indices = indices[split:]
    val_indices = indices[:split]
    
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)
    
    train_loader = DataLoader(
        dataset, 
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=val_sampler,
        num_workers=4,
        pin_memory=True
    )
    
    print(f"Training samples: {len(train_indices)}, Validation samples: {len(val_indices)}")
    
    # Create model
    model = UNet(n_channels=3, n_classes=1, bilinear=True).to(device)
    
    # Loss function
    criterion = nn.BCEWithLogitsLoss()
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'max', patience=3, factor=0.1, verbose=True
    )
    
    # Train model
    history, saved_model_path = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=num_epochs,
        device=device,
        output_dir=output_dir
    )
    
    # Load saved model
    model.load_state_dict(torch.load(saved_model_path))
    
    # Visualize predictions
    visualize_predictions(model, dataset, num_samples=5, device=device, 
                         output_dir=output_dir, project_root=project_root)
    
    # Plot training history
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['val_iou'], label='Validation IoU')
    plt.xlabel('Epoch')
    plt.ylabel('IoU')
    plt.title('Validation IoU')
    plt.legend()
    
    plt.tight_layout()
    
    # Save to output directory
    plt.savefig(os.path.join(output_dir, 'training_history.png'))
    
    # Also save to project root for easy access
    plt.savefig(os.path.join(project_root, 'unet_learning_curves.png'))
    
    plt.close()
    
    print("Training completed!")
    print(f"Learning curves saved to {os.path.join(project_root, 'unet_learning_curves.png')}")


if __name__ == "__main__":
    main()