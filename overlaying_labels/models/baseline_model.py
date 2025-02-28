import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet50_Weights

class ShallowCNN(nn.Module):
    """
    A shallow CNN branch (Figure 10, left side) for the pre-disaster image.
    We'll assume input size is (3 x 128 x 128). If your input size differs,
    adjust layers accordingly.
    """
    def __init__(self):
        super().__init__()
        
        # 4 blocks: each has a 5x5 conv (with padding=2), ReLU, 2x2 max pool.
        # Channel progression: 3 -> 16 -> 32 -> 64 -> 128
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        # After 4 pooling ops, 128x128 -> 8x8 => shape (batch, 128, 8, 8) => 8192 features.
        # Then a linear layer to reduce to 512 dims.
        self.fc = nn.Sequential(
            nn.Linear(128 * 8 * 8, 512),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        """
        x shape expected: (batch_size, 3, 128, 128)
        """
        x = self.features(x)              # => (batch, 128, 8, 8)
        x = x.view(x.size(0), -1)         # Flatten => (batch, 8192)
        x = self.fc(x)                    # => (batch, 512)
        return x


class BaselineModel(nn.Module):
    """
    Two-branch model from the xBD paper (Figure 10):
    - Branch 1: Pretrained ResNet50 (for post-disaster image).
    - Branch 2: Shallow CNN (for pre-disaster image).
    - Concat the features, then pass through dense layers for classification.
    """
    def __init__(self, num_classes=4):
        super().__init__()

        # 1) Load ResNet50 with IMAGENET1K_V2 weights
        #    This sets up the official improved training recipe weights
        self.resnet = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        
        # Remove ResNet's final fc layer, to get a 2048-d feature vector
        self.resnet.fc = nn.Identity()

        # 2) Shallow CNN branch
        self.shallow_cnn = ShallowCNN()  # outputs 512-d features

        # 3) Final classifier
        #    After concatenation, we have 2048 (from ResNet) + 512 (from ShallowCNN) = 2560
        #    Then we apply Dense(224) -> Dense(224) -> Output(num_classes)
        self.classifier = nn.Sequential(
            nn.Linear(2048 + 512, 224),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            
            nn.Linear(224, 224),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            
            nn.Linear(224, num_classes)
        )

    def forward(self, pre_img, post_img):
        """
        pre_img : (batch_size, 3, 128, 128)   => Shallow CNN
        post_img: (batch_size, 3, 224, 224)  => ResNet50

        Return: (batch_size, num_classes)
        """
        # Post-disaster image through ResNet50
        post_feats = self.resnet(post_img)   # => (batch, 2048)

        # Pre-disaster image through Shallow CNN
        pre_feats = self.shallow_cnn(pre_img)  # => (batch, 512)

        # Concatenate features
        combined = torch.cat([post_feats, pre_feats], dim=1)  # => (batch, 2560)

        # Classify
        out = self.classifier(combined)  # => (batch, num_classes)
        return out


if __name__ == "__main__":
    # Quick shape test
    model = BaselineModel(num_classes=4)
    print(model)

    # Create dummy data
    pre_img_dummy = torch.randn(2, 3, 128, 128)
    post_img_dummy = torch.randn(2, 3, 224, 224)

    outputs = model(pre_img_dummy, post_img_dummy)
    print("Output shape:", outputs.shape)  # Expected: [2, 4]