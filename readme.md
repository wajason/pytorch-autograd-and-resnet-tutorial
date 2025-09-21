# 🚀 PyTorch Autograd & ResNet Tutorial

[![Stars](https://img.shields.io/github/stars/yourname/yourrepo?style=social)](https://github.com/yourname/yourrepo/stargazers)
[![License](https://img.shields.io/github/license/yourname/yourrepo)](LICENSE)

This repository is a **hands-on deep learning tutorial** that takes you from **PyTorch Autograd basics** to building **CNNs and ResNet** for **CIFAR-10 classification**.

It’s designed for learners who want to **understand how PyTorch really works under the hood** instead of just calling high-level APIs.  
Every block is implemented step by step — from barebone tensor ops to modular ResNet.

---

## 📌 Features
- ✅ **Part I: Autograd** — understand PyTorch’s automatic differentiation.
- ✅ **Part II: Barebones ConvNet** — implement convolution layers manually with tensors.
- ✅ **Part III: nn.Module API** — build a reusable `ThreeLayerConvNet`.
- ✅ **Part IV: Sequential API** — compact model definitions with `nn.Sequential`.
- ✅ **Part V: ResNet for CIFAR-10** — implement `PlainBlock`, `ResidualBlock`, and `BottleneckBlock`.

---

## 📂 Project Structure
.
```bash
├── helper.py # Utility functions for training/evaluation
├── pytorch_autograd_and_nn.py # Core implementations of CNN & ResNet
├── Pytorch Autograd and NN.ipynb # Jupyter Notebook walkthrough
└── README.md
```


---

## 🚀 Quick Start
Clone this repo and run the notebook:

```bash
git clone https://github.com/wajason/pytorch-autograd-and-resnet-tutorial.git
cd pytorch-autograd-and-resnet-tutorial
jupyter notebook "Pytorch Autograd and NN.ipynb"

## 🎯 Learning Goals
Deep dive into PyTorch Autograd

Learn ConvNet internals (weights, bias, forward pass)

Understand skip connections & why ResNet works

Gain hands-on experience building deep CNNs from scratch
