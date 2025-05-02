# Python packages
from termcolor import colored
from typing import Dict
import copy

# PyTorch & Pytorch Lightning
from lightning.pytorch import LightningModule
from lightning.pytorch.loggers.wandb import WandbLogger
from torch import nn
from torchvision import models
from torchvision.models.alexnet import AlexNet
from torchvision.models.resnet import ResNet, BasicBlock, Bottleneck
import torch
import torch.nn.functional as F

# Custom packages
from src.metric import MyAccuracy, MyF1Score
import src.config as cfg
from src.util import show_setting

# [TODO: Optional] Rewrite this class if you want

# class MyNetwork(AlexNet):
#     def __init__(self, num_classes, dropout):
#         super().__init__(
#             num_classes = num_classes,
#             dropout = dropout
#         )
#         # [TODO] Modify feature extractor part in AlexNet
#         self.features = nn.Sequential(
#             nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),  # 64x64 → 32x32
#             nn.BatchNorm2d(64),
#             nn.GELU(),

#             nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(128),
#             nn.GELU(),

#             nn.Conv2d(128, 192, kernel_size=3, stride=2, padding=1),  # 32x32 → 16x16
#             nn.BatchNorm2d(192),
#             nn.GELU(),

#             nn.Conv2d(192, 256, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(256),
#             nn.GELU(),

#             nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),  # 16x16 → 8x8
#             nn.BatchNorm2d(256),
#             nn.GELU(),
#         )

#         self.avgpool = nn.AdaptiveAvgPool2d((4, 4))

#         self.classifier = nn.Sequential(
#             nn.Dropout(p=dropout),
#             nn.Linear(256 * 4 * 4, 1024),
#             nn.GELU(),
#             nn.Dropout(p=dropout),
#             nn.Linear(1024, 1024),
#             nn.GELU(),
#             nn.Linear(1024, num_classes),
#         )
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         # [TODO: Optional] Modify this as well if you want
#         x = self.features(x)
#         x = self.avgpool(x)
#         x = torch.flatten(x, 1)
#         x = self.classifier(x)
#         return x

class MyNetwork(nn.Module):
    def __init__(self, num_classes, dropout):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),  # 64→32
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Dropout2d(p=dropout)  # ✅ conv dropout
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # 32→16
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Dropout2d(p=dropout)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  # 16→8
            nn.BatchNorm2d(256),
            nn.GELU()
        )

        # Skip connection (projection to match channel/size)
        self.skip1 = nn.Conv2d(64, 128, kernel_size=1, stride=2)
        self.skip2 = nn.Conv2d(128, 256, kernel_size=1, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((4, 4))

        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(256 * 4 * 4, 512),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x1 = self.conv1(x)                        # (512, 64, 32, 32)
        x2 = self.conv2(x1) + self.skip1(x1)      # (512, 128, 16, 16)
        x3 = self.conv3(x2) + self.skip2(x2)      # (512, 256, 8, 8)

        x = self.avgpool(x3)                      # (512, 256, 4, 4)
        x = torch.flatten(x, 1)                   # (512, 4096)
        x = self.classifier(x)                    # (512, num_classes)
        return x


class SimpleClassifier(LightningModule):
    def __init__(self,
                 model_name: str = 'resnet18',
                 num_classes: int = 200,
                 optimizer_params: Dict = dict(),
                 scheduler_params: Dict = dict(),
                 dropout: float = 0.5,
        ):
        super().__init__()

        # Network
        if model_name == 'MyNetwork':
            self.model = MyNetwork(num_classes, dropout)
        else:
            models_list = models.list_models()
            assert model_name in models_list, f'Unknown model name: {model_name}. Choose one from {", ".join(models_list)}'
            self.model = models.get_model(model_name, num_classes=num_classes, dropout=dropout)

        # Loss function
        self.loss_fn = nn.CrossEntropyLoss()

        # Metric
        self.accuracy = MyAccuracy()
        self.f1score = MyF1Score(num_classes=num_classes)

        # Hyperparameters
        self.save_hyperparameters()

    def on_train_start(self):
        show_setting(cfg)

    def configure_optimizers(self):
        optim_params = copy.deepcopy(self.hparams.optimizer_params)
        optim_type = optim_params.pop('type')
        optimizer = getattr(torch.optim, optim_type)(self.parameters(), **optim_params)

        scheduler_params = copy.deepcopy(self.hparams.scheduler_params)
        scheduler_type = scheduler_params.pop('type')

        if scheduler_type == 'OneCycleLR':
            scheduler = {
                'scheduler': torch.optim.lr_scheduler.OneCycleLR(
                    optimizer,
                    max_lr=scheduler_params.pop('max_lr'),
                    steps_per_epoch=self.trainer.estimated_stepping_batches // self.trainer.max_epochs,
                    **scheduler_params
                ),
                'interval': 'step',
                'frequency': 1
            }
        else:
            scheduler = {
                'scheduler': getattr(torch.optim.lr_scheduler, scheduler_type)(optimizer, **scheduler_params),
                'interval': 'epoch',
                'frequency': 1
            }

        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        loss, scores, y = self._common_step(batch)
        accuracy = self.accuracy(scores, y)
        self.log_dict({'loss/train': loss, 'accuracy/train': accuracy},
                      on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, scores, y = self._common_step(batch)
        accuracy = self.accuracy(scores, y)
        self.f1score.update(scores, y)
        self.log_dict({'loss/val': loss, 'accuracy/val': accuracy},
                      on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self._wandb_log_image(batch, batch_idx, scores, frequency = cfg.WANDB_IMG_LOG_FREQ)

    def on_validation_epoch_end(self):
        f1_per_class = self.f1score.compute()
        macro_f1 = f1_per_class.mean()
        self.log("f1/val_macro", macro_f1, prog_bar=True, logger=True)

    def _common_step(self, batch):
        x, y = batch
        scores = self.forward(x)
        loss = self.loss_fn(scores, y)
        return loss, scores, y

    def _wandb_log_image(self, batch, batch_idx, preds, frequency = 100):
        if not isinstance(self.logger, WandbLogger):
            if batch_idx == 0:
                self.print(colored("Please use WandbLogger to log images.", color='blue', attrs=('bold',)))
            return

        if batch_idx % frequency == 0:
            x, y = batch
            preds = torch.argmax(preds, dim=1)
            self.logger.log_image(
                key=f'pred/val/batch{batch_idx:5d}_sample_0',
                images=[x[0].to('cpu')],
                caption=[f'GT: {y[0].item()}, Pred: {preds[0].item()}'])
