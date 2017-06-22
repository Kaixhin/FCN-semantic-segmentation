FCN-semantic-segmentation
=========================

Simple end-to-end semantic segmentation using fully convolutional networks [[1]](#references). Takes a pretrained 34-layer ResNet [[2]](#references), removes the fully connected layers, and adds transposed convolution layers with skip residual connections from lower layers. Initialises upsampling convolutions with bilinear interpolation filters and zeros the final (classification) layer. Uses an independent cross-entropy loss per class.

Calculates and plots class-wise and mean intersection-over-union. Checkpoints the network every epoch.

**Note: This code does not achieve great results (achieves ~40 IoU fairly quickly, but converges there). Contributions to fix this are welcome! The goal of this repo is to provide strong, simple and efficient baselines for semantic segmentation using the FCN method, so this shouldn't be restricted to using ResNet 34 etc.**

Requirements
------------

- [CUDA](https://developer.nvidia.com/cuda-zone)
- [PyTorch](http://pytorch.org/)
- [matplotlib](https://matplotlib.org/)
- [Cityscapes Dataset](https://www.cityscapes-dataset.com/)

References
----------

[1] [Fully convolutional networks for semantic segmentation](https://arxiv.org/abs/1605.06211)  
[2] [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)  
