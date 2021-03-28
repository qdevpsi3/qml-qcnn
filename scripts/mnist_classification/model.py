import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pytorch_lightning.metrics.functional import accuracy
from qcnn import QuantumConv2d  # import convolutional layer

from data import get_loader


class QuantumConvNet(pl.LightningModule):
    def __init__(self, hparams):
        super(QuantumConvNet, self).__init__()
        self.hparams = hparams
        self.build_q_nn()

    def build_q_nn(self):

        # get quantum hyperparameters
        eps = self.hparams.quantum_eps
        cap = self.hparams.quantum_cap
        ratio = self.hparams.quantum_ratio
        delta = self.hparams.quantum_delta

        # build neural network classifier
        self.qonv1 = QuantumConv2d(1, 5, 7, eps, cap, ratio, delta, stride=1)
        self.qonv2 = QuantumConv2d(5, 10, 7, eps, cap, ratio, delta, stride=1)
        self.fc1 = nn.Linear(2560, 300)
        self.fc2 = nn.Linear(300, 50)
        self.fc3 = nn.Linear(50, 10)

    def forward(self, x):
        x = self.qonv1(x)
        x = self.qonv2(x)
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.log_softmax(self.fc3(x), dim=0)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y)
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        return loss

    def configure_optimizers(self):
        lr, momentum = self.hparams.lr, self.hparams.momentum
        optimizer = optim.SGD(self.parameters(), lr, momentum)
        return optimizer

    def train_dataloader(self):
        batch_size = self.hparams.batch_size
        return get_loader(is_train=True, batch_size=batch_size)

    def val_dataloader(self):
        batch_size = self.hparams.batch_size
        return get_loader(is_train=False, batch_size=batch_size)
