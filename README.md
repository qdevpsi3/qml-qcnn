<h1 align="center" style="margin-top: 0px;"> <b>Quantum Algorithms for Deep Convolutional Neural Networks</b></h1>
<div align="center" >

[![paper](https://img.shields.io/static/v1.svg?label=Paper&message=KLP19&color=blue)](https://arxiv.org/abs/1911.01117)
[![framework](https://img.shields.io/static/v1.svg?label=Framework&message=PyTorch&color=ee4c2d)](https://pytorch.org)
[![packages](https://img.shields.io/static/v1.svg?label=Made%20with&message=PyTorch%20Lightning&color=blueviolet)](https://www.pytorchlightning.ai)
[![license](https://img.shields.io/static/v1.svg?label=License&message=GPL%20v3.0&color=blue)](https://www.gnu.org/licenses/gpl-3.0.html)
[![exp](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/qdevpsi3/qml-qcnn/blob/main/notebooks/mnist_classification.ipynb)
</div>

## Description
This repository contains an implementation of the <ins>quantum convolutional layer</ins> and its application to the  <ins>MNIST classification task</ins> in :

- Paper : **Quantum Algorithms for Deep Convolutional Neural Networks**
- Authors : **Kerenidis, Landman and Prakash**
- Date : **2019**

## Setup
To install the <ins>quantum convolutional layer</ins>, clone this repository and execute the following commands :

```
$ cd qml-qcnn
$ pip install -e .
```

## Details
The quantum convolutional layer is a <ins>PyTorch module</ins> with the following parameters : 
- **in_channels** (int): *number of input channels.*
- **out_channels** (int): *number of output channels.*
- **kernel_size** (int): *size of the convolution kernel.*
- <span style="color:red;">**eps** (float):</span> *precision of quantum multiplication.*
- <span style="color:red;">**cap** (float):</span> *value for cap 'relu' activation function.*
- <span style="color:red;">**ratio** (float):</span> *precision of quantum tomography.*
- <span style="color:red;">**delta** (float):</span> *precision of quantum gradient estimation.*
- **stride** (int, optional): *convolution stride.* Defaults to 1.
- **padding** (int, optional): *convolution padding.* Defaults to 0.
- **dilation** (int, optional): *convolution dilation.* Defaults to 1.

The bias is unsupported for now. 
## Example
In Pytorch, you can use the 2D convolution module as follows : 
```python
from torch import nn

torch_conv2d = nn.Conv2d(16, 32, 3, stride=1)
```
Similarly, you can use the 2D <ins>quantum</ins> convolution module as follows :
```python
import qcnn

eps = 0.01
cap = 10.
ratio = 0.5
delta = 0.01

torch_conv2d = qcnn.QuantumConv2d(16, 32, 3, eps, cap, ratio, delta, stride=1)
```

## Experiments
The <ins>experiments</ins> in the paper are reproduced for <ins>MNIST classification task</ins> using *PyTorch Lightning*. 

- Option 1 : Open in [Colab](https://colab.research.google.com/github/qdevpsi3/qml-qcnn/blob/main/notebooks/mnist_classification.ipynb). You can activate the <ins>GPU</ins> in *Notebook Settings*.
- Option 2 : Run on local machine. First, you need to install *PyTorch Lightning* :
```
$ pip install pytorch_lightning
```
You can run an experiment using the following command :
```
$ cd qml-qcnn/scripts/
$ python mnist_classification --eps=0.01 --cap=10 --ratio=0.5 --delta=0.01
```

## Acknowledgements
- [Official implementation](https://github.com/JonasLandman/QCNN)
