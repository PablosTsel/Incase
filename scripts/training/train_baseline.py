#!/usr/bin/env python3
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler, Subset
from tqdm import tqdm
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import f1_score
import random
from datetime import datetime
import time
import torch.nn.functional as F

# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(project_root)

from overlaying_labels.models.baseline_model import BaselineModel
from options.utils.data_loader import XBDPatchDataset, DAMAGE_LABELS

def seed_everything(seed=42):
    import random, numpy as np, torch, os
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"Random seed set to {seed}")

def compute_sample_weights(dataset, indices, weight_scale=1.0):
    """Compute sample weights for the given dataset indices"""
    # Extract labels for the given indices
    labels = [dataset.samples[i]["label"] for i in indices]
    counts = Counter(labels)
    total = sum(counts.values())
    num_classes = len(DAMAGE_LABELS)
    
    # Compute inverse frequency weights
    class_weights = {cls: (total / (num_classes * counts[cls]))**weight_scale for cls in counts}
    
    # Create sample weights
    sample_weights = [class_weights[label] for label in labels]
    
    return sample_weights, class_weights

def mixup_data(x1, x2, y, alpha=0.2, device='cuda'):
    """Applies mixup augmentation to the data"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x1.size()[0]
    index = torch.randperm(batch_size).to(device)

    mixed_x1 = lam * x1 + (1 - lam) * x1[index, :]
    mixed_x2 = lam * x2 + (1 - lam) * x2[index, :]
    y_a, y_b = y, y[index]
    
    return mixed_x1, mixed_x2, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Criterion for mixup training"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def plot_learning_curves(epochs, train_losses, val_losses, val_accuracies, per_class_f1, class_names, save_path):
    plt.figure(figsize=(16, 12))
    
    # Plot loss curves
    plt.subplot(2, 2, 1)
    plt.plot(epochs, train_losses, label="Train Loss", marker='o', color='blue')
    plt.plot(epochs, val_losses, label="Val Loss", marker='o', color='red')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Plot validation accuracy
    plt.subplot(2, 2, 2)
    plt.plot(epochs, val_accuracies, label="Val Accuracy (%)", marker='o', color='green')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Validation Accuracy")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Plot per-class F1 scores
    plt.subplot(2, 2, 3)
    per_class_f1 = np.array(per_class_f1)  # shape: (num_epochs, num_classes)
    for i, cls_name in enumerate(class_names):
        plt.plot(epochs, per_class_f1[:, i], label=f"{cls_name}", marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("F1 Score")
    plt.title("Per-Class F1 Scores")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Plot class distribution of last batch
    plt.subplot(2, 2, 4)
    plt.bar(class_names, [per_class_f1[-1, i] for i in range(len(class_names))])
    plt.xlabel("Class")
    plt.ylabel("Final F1 Score")
    plt.title("Final F1 Score by Class")
    plt.xticks(rotation=45)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Learning curves plot saved to {save_path}")


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha  # Weight for each class
        self.reduction = reduction

    def forward(self, inputs, targets):
        # Use nn.functional directly instead of F
        ce_loss = torch.nn.functional.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss)
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


def main():
    start_time = time.time()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    seed_everything(42)
    
    # Hyperparameters & settings
    root_dir = os.path.join(project_root, "data", "xBD")
    batch_size = 32
    lr = 0.0001
    num_epochs = 8
    val_ratio = 0.15
    use_focal_loss = True
    use_mixup = False  # Disable mixup for now
    weight_scale = 0.7
    
    # Create output directories
    output_dir = os.path.join(project_root, "output", f"training_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    model_dir = os.path.join(output_dir, "models")
    os.makedirs(model_dir, exist_ok=True)
    pictures_dir = os.path.join(output_dir, "pictures")
    os.makedirs(pictures_dir, exist_ok=True)
    
    # Save configuration
    config = {
        "timestamp": timestamp,
        "batch_size": batch_size,
        "learning_rate": lr,
        "num_epochs": num_epochs,
        "val_ratio": val_ratio,
        "use_focal_loss": use_focal_loss,
        "use_mixup": use_mixup,
        "weight_scale": weight_scale,
    }
    
    with open(os.path.join(output_dir, "config.txt"), "w") as f:
        for key, value in config.items():
            f.write(f"{key}: {value}\n")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print("Initializing XBDPatchDataset for training...")
    full_dataset = XBDPatchDataset(
        root_dir=root_dir,
        pre_crop_size=128,
        post_crop_size=224,
        use_xy=True,
        max_samples=None
    )
    total_samples = len(full_dataset)
    print(f"Total samples: {total_samples}")

    # Create stratified train-val split directly
    def create_stratified_split(dataset, val_ratio=0.15, seed=42):
        random.seed(seed)
        np.random.seed(seed)
        
        # Group samples by label
        label_to_indices = {}
        for idx, sample in enumerate(dataset.samples):
            label = sample['label']
            if label not in label_to_indices:
                label_to_indices[label] = []
            label_to_indices[label].append(idx)
        
        train_indices = []
        val_indices = []
        
        # Stratified sampling
        for label, indices in label_to_indices.items():
            random.shuffle(indices)
            val_size = int(len(indices) * val_ratio)
            val_indices.extend(indices[:val_size])
            train_indices.extend(indices[val_size:])
        
        random.shuffle(train_indices)
        random.shuffle(val_indices)
        
        return train_indices, val_indices
    
    # Create the split
    train_indices, val_indices = create_stratified_split(full_dataset, val_ratio, seed=42)
    print(f"Training samples: {len(train_indices)}, Validation samples: {len(val_indices)}")

    # Compute class weights for weighted sampling
    sample_weights, class_weights = compute_sample_weights(full_dataset, train_indices, weight_scale)
    sampler = SubsetRandomSampler(train_indices)
    
    # Create train and validation datasets
    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)
    
    # Compute sample weights for weighted sampling
    sample_weights, class_weights = compute_sample_weights(full_dataset, train_indices, weight_scale)
    sampler = torch.utils.data.WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        sampler=sampler, 
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size,
        shuffle=False,  # No need for sampler with validation set
        num_workers=4,
        pin_memory=True
    )

    # Initialize model - check if pretrained is a valid parameter
    try:
        model = BaselineModel(num_classes=4, pretrained=True, dropout_rate=0.5).to(device)
    except TypeError:
        # If pretrained is not a valid parameter, try without it
        print("Warning: 'pretrained' parameter not supported, using default initialization")
        model = BaselineModel(num_classes=4).to(device)
    
    # Define class weights for loss function
    weight_tensor = torch.tensor([class_weights.get(i, 1.0) for i in range(4)], dtype=torch.float).to(device)
    
    # Loss function
    if use_focal_loss:
        criterion = FocalLoss(alpha=weight_tensor, gamma=2.0, reduction='mean')
        print("Using Focal Loss with gamma=2.0")
    else:
        criterion = nn.CrossEntropyLoss(weight=weight_tensor)
        print("Using Weighted CrossEntropyLoss")
    
    # Optimizer with weight decay
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=lr,
        epochs=num_epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.3,
        anneal_strategy='cos'
    )

    # Training metrics tracking
    epochs_list = []
    train_losses = []
    val_losses = []
    val_accuracies = []
    per_class_f1_scores = []
    best_f1_score = 0.0
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        epoch_start_time = time.time()
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [TRAIN]", leave=True)
        for pre_batch, post_batch, labels in train_pbar:
            pre_batch = pre_batch.to(device)
            post_batch = post_batch.to(device)
            labels = labels.to(device)
            
            # Apply mixup if enabled
            if use_mixup and np.random.random() < 0.5:
                pre_batch, post_batch, labels_a, labels_b, lam = mixup_data(pre_batch, post_batch, labels, alpha=0.2, device=device)
                mixup_applied = True
            else:
                mixup_applied = False

            optimizer.zero_grad()
            outputs = model(pre_batch, post_batch)
            
            if mixup_applied:
                loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
            else:
                loss = criterion(outputs, labels)
                
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            running_loss += loss.item()
            train_pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            
        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        val_loss_epoch = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [VAL]", leave=True)
            for pre_batch, post_batch, labels in val_pbar:
                pre_batch = pre_batch.to(device)
                post_batch = post_batch.to(device)
                labels = labels.to(device)
                
                outputs = model(pre_batch, post_batch)
                loss = criterion(outputs, labels)
                val_loss_epoch += loss.item()
                
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                val_pbar.set_postfix({"loss": f"{loss.item():.4f}"})
                
        val_loss_epoch /= len(val_loader)
        val_losses.append(val_loss_epoch)
        val_acc = 100 * correct / total if total > 0 else 0
        val_accuracies.append(val_acc)
        
        # Calculate F1 scores
        epoch_f1 = f1_score(all_labels, all_preds, average=None, labels=[0, 1, 2, 3])
        per_class_f1_scores.append(epoch_f1)
        
        # Calculate macro F1 for model saving
        macro_f1 = np.mean(epoch_f1)
        epochs_list.append(epoch + 1)
        
        # Calculate elapsed time
        epoch_time = time.time() - epoch_start_time
        
        print(f"Epoch [{epoch+1}/{num_epochs}] "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {val_loss_epoch:.4f} | "
              f"Val Acc: {val_acc:.2f}% | "
              f"Macro F1: {macro_f1:.4f} | "
              f"Time: {epoch_time:.1f}s")
        
        print(f"Per-class F1 Scores: {', '.join([f'{c}: {f:.4f}' for c, f in zip(DAMAGE_LABELS.keys(), epoch_f1)])}")
        
        # Save the model if it's the best so far
        if macro_f1 > best_f1_score:
            best_f1_score = macro_f1
            best_model_path = os.path.join(model_dir, f"best_model_epoch_{epoch+1}.pt")
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved with Macro F1: {best_f1_score:.4f}")

    # Save the final model regardless
    final_save_path = os.path.join(project_root, "baseline_model_final.pt")
    torch.save(model.state_dict(), final_save_path)
    print(f"Final model saved to {final_save_path}")
    
    # Create learning curves plot
    plot_save_path = os.path.join(pictures_dir, "learning_curves.png")
    plot_learning_curves(
        epochs_list, 
        train_losses, 
        val_losses, 
        val_accuracies, 
        per_class_f1_scores,
        list(DAMAGE_LABELS.keys()),
        plot_save_path
    )
    
    # Save all training metrics
    metrics = {
        "epochs": epochs_list,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "val_accuracies": val_accuracies,
        "per_class_f1_scores": per_class_f1_scores,
    }
    
    metrics_path = os.path.join(output_dir, "training_metrics.txt")
    with open(metrics_path, "w") as f:
        for key, values in metrics.items():
            if key == "per_class_f1_scores":
                f.write(f"{key}:\n")
                for epoch_idx, epoch_scores in enumerate(values):
                    f.write(f"  Epoch {epoch_idx+1}: {epoch_scores}\n")
            else:
                f.write(f"{key}: {values}\n")
    
    total_time = time.time() - start_time
    print(f"Training completed in {total_time/60:.2f} minutes")
    print(f"Best validation Macro F1: {best_f1_score:.4f}")
    print(f"All training artifacts saved to {output_dir}")
    print("Run 'evaluate_baseline.py' to test the model")

if __name__ == "__main__":
    main()