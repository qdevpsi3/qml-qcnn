import argparse
import warnings

warnings.filterwarnings('ignore')

import pytorch_lightning as pl
import torch

from model import QuantumConvNet

parser = argparse.ArgumentParser()
parser.add_argument('--eps',
                    type=float,
                    default=0.01,
                    help='precision of quantum multiplication')
parser.add_argument('--cap',
                    type=float,
                    default=10.,
                    help='value of cap relu activation function')
parser.add_argument('--ratio',
                    type=float,
                    default=0.5,
                    help='precision of quantum tomography')
parser.add_argument('--delta',
                    type=float,
                    default=0.01,
                    help='precision of quantum gradient estimation')
parser.add_argument('--lr',
                    type=float,
                    default=0.01,
                    help='learning rate value')
parser.add_argument('--momentum',
                    type=float,
                    default=0.5,
                    help='gradient momentum')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
hparams = parser.parse_args()

print('Hyperpameters')
for param in vars(hparams):
    print('* ', param, ' : ', getattr(hparams, param))

model = QuantumConvNet(hparams)
use_gpu = int(torch.cuda.is_available())
trainer = pl.Trainer(gpus=use_gpu,
                     max_epochs=3,
                     progress_bar_refresh_rate=20,
                     default_root_dir='mnist_classification/')
trainer.fit(model)
