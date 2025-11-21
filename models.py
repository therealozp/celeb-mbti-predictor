from torchvision.models import resnet18, ResNet18_Weights
import torch.nn as nn
import torch


class MBTIMultiHeadAffectNetPretrained(nn.Module):
    def __init__(self, freeze_backbone=False):
        super().__init__()

        base_model = resnet18()

        num_ftrs = base_model.fc.in_features
        base_model.fc = nn.Linear(num_ftrs, 8)
        base_model.load_state_dict(torch.load("resnet18_affectnet_best.pth"))

        # Remove the final fully connected layer (fc)
        # Output of this sequential block will be [Batch, 512, 1, 1]
        self.backbone = nn.Sequential(*list(base_model.children())[:-1])

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.head_ie = self._make_head()
        self.head_ns = self._make_head()
        self.head_tf = self._make_head()
        self.head_jp = self._make_head()

    def _make_head(self):
        """
        Creates a small sub-network for a specific trait.
        Includes Dropout to fight your overfitting problem.
        """
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),  # Helps stability
            nn.ReLU(),
            nn.Dropout(0.5),  # Critical for your small dataset
            nn.Linear(128, 1),  # Output 1 logit
        )

    def forward(self, x):
        # Extract shared features
        features = self.backbone(x)  # Shape: [Batch, 512, 1, 1]

        # Run features through each specific head
        out_ie = self.head_ie(features)
        out_ns = self.head_ns(features)
        out_tf = self.head_tf(features)
        out_jp = self.head_jp(features)

        # Concatenate them to match target shape [Batch, 4]
        # This makes it compatible with your existing loop/Loss function
        return torch.cat([out_ie, out_ns, out_tf, out_jp], dim=1)
