import torch
import torch.nn as nn

class EvolvingFERModel(nn.Module):
    def __init__(self, stage='stage1_tiny', num_classes=7):
        super(EvolvingFERModel, self).__init__()
        self.stage = stage
        
        if stage == 'stage1_tiny':
            self.features = nn.Sequential(
                nn.Conv2d(1, 12, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2, 2)
            )
            self.classifier = nn.Linear(12 * 24 * 24, num_classes)
            
        elif stage == 'stage2_baseline':
            self.features = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(16, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2, 2)
            )
            self.classifier = nn.Linear(32 * 12 * 12, num_classes)
            
        elif stage == 'stage3_deep_unregularized':
            self.features = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2, 2)
            )
            self.classifier = nn.Sequential(
                nn.Linear(128 * 12 * 12, 256),
                nn.ReLU(),
                nn.Linear(256, num_classes)
            )
            
        elif stage == 'stage4_regularized':
            self.features = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.MaxPool2d(2, 2),
                
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.MaxPool2d(2, 2),
                
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.Dropout(0.4),
                nn.MaxPool2d(2, 2)
            )
            self.classifier = nn.Sequential(
                nn.Linear(128 * 6 * 6, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(0.4),
                nn.Linear(256, num_classes)
            )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)