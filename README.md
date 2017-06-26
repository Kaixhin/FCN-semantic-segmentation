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

Instructions
------------

1. Install all of the required software. To feasibly run the training, CUDA is needed. The crop size and batch size can be tailored to your GPU memory (the default crop and batch sizes use ~10GB of GPU RAM).
2. Register on the Cityscapes website to [access the dataset](https://www.cityscapes-dataset.com/downloads/).
3. Download and extract the training/validation RGB data (`leftImg8bit_trainvaltest`) and ground truth data (`gtFine_trainvaltest`).
4. Run `python main.py <options>`.

First a Dataset object is set up, returning the RGB inputs, one-hot targets (for independent classification) and label targets. During training, the images are randomly cropped and horizontally flipped. Testing calculates IoU scores and produces a subset of coloured predictions that match the coloured ground truth.

References
----------

[1] [Fully convolutional networks for semantic segmentation](https://arxiv.org/abs/1605.06211)  
[2] [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)  
