import pytorch_lightning as pl
import torchmetrics
import torch
import torch.nn.functional as F


class Wrapper(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.running_loss_train = 0
        self.running_loss_val = 0
        self.accuracy_metric_train = torchmetrics.Accuracy()
        self.accuracy_metric_val = torchmetrics.Accuracy()

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

    def compute_loss(self, logits, ground_truth):
        return F.cross_entropy(logits, ground_truth)

    def training_step(self, train_batch, *args, **kwargs):
        x, y = train_batch
        outputs = self.forward(x)
        loss = self.compute_loss(outputs, y)
        self.running_loss_train += loss.item()
        self.accuracy_metric_train(outputs, y)
        return loss

    def training_step_end(self, *args, **kwargs):
        self.log("train/accuracy", self.accuracy_metric_train)
        self.log("train/running_loss", self.running_loss_train)
        self.running_loss_train = 0
        self.accuracy_metric_train.reset()

    def validation_step(self, val_batch, batch_index, *args, **kwargs):
        x, y = val_batch
        outputs = self.forward(x)
        loss = self.compute_loss(outputs, y)
        self.running_loss_val += loss.item()
        self.accuracy_metric_val(outputs, y)

    def validation_step_end(self, *args, **kwargs):
        self.log("val/accuracy", self.accuracy_metric_val)
        self.log("val/running_loss", self.running_loss_val)
        self.running_loss_val = 0
        self.accuracy_metric_val.reset()
