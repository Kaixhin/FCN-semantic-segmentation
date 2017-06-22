from argparse import ArgumentParser
import os
import random
import torch
from PIL import Image
from torch import optim
from torch import nn
from torch.nn import init
from torch.nn import functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from torchvision.models.resnet import BasicBlock, ResNet
from torchvision.utils import save_image
from matplotlib import pyplot as plt


# Setup
parser = ArgumentParser(description='Semantic segmentation')
parser.add_argument('--seed', type=int, default=42, help='Random seed')
parser.add_argument('--workers', type=int, default=8, help='Data loader workers')
parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
parser.add_argument('--crop-size', type=int, default=512, help='Training crop size')
parser.add_argument('--lr', type=float, default=5e-5, help='Learning rate')
parser.add_argument('--weight-decay', type=float, default=2e-4, help='Weight decay')
parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
args = parser.parse_args()
random.seed(args.seed)
torch.manual_seed(args.seed)
if not os.path.exists('results'):
  os.makedirs('results')
plt.switch_backend('agg')  # Allow plotting when running remotely

# Labels: -1 license plate, 0 unlabeled, 1 ego vehicle, 2 rectification border, 3 out of roi, 4 static, 5 dynamic, 6 ground, 7 road, 8 sidewalk, 9 parking, 10 rail track, 11 building, 12 wall, 13 fence, 14 guard rail, 15 bridge, 16 tunnel, 17 pole, 18 polegroup, 19 traffic light, 20 traffic sign, 21 vegetation, 22 terrain, 23 sky, 24 person, 25 rider, 26 car, 27 truck, 28 bus, 29 caravan, 30 trailer, 31 train, 32 motorcycle, 33 bicycle
num_classes = 20
full_to_train = {-1: 19, 0: 19, 1: 19, 2: 19, 3: 19, 4: 19, 5: 19, 6: 19, 7: 0, 8: 1, 9: 19, 10: 19, 11: 2, 12: 3, 13: 4, 14: 19, 15: 19, 16: 19, 17: 5, 18: 19, 19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14, 28: 15, 29: 19, 30: 19, 31: 16, 32: 17, 33: 18}
train_to_full = {0: 7, 1: 8, 2: 11, 3: 12, 4: 13, 5: 17, 6: 19, 7: 20, 8: 21, 9: 22, 10: 23, 11: 24, 12: 25, 13: 26, 14: 27, 15: 28, 16: 31, 17: 32, 18: 33, 19: 0}
full_to_colour = {0: (0, 0, 0), 7: (128, 64, 128), 8: (244, 35, 232), 11: (70, 70, 70), 12: (102, 102, 156), 13: (190, 153, 153), 17: (153, 153, 153), 19: (250, 170, 30), 20: (220, 220, 0), 21: (107, 142, 35), 22: (152, 251, 152), 23: (70, 130, 180), 24: (220, 20, 60), 25: (255, 0, 0), 26: (0, 0, 142), 27: (0, 0, 70), 28: (0, 60,100), 31: (0, 80, 100), 32: (0, 0, 230), 33: (119, 11, 32)}


# Data
class CityscapesDataset(Dataset):
  def __init__(self, split='train', crop=None, flip=False):
    super().__init__()
    self.crop = crop
    self.flip = flip
    self.inputs = []
    self.targets = []

    for root, _, filenames in os.walk(os.path.join('leftImg8bit_trainvaltest', 'leftImg8bit', split)):
      for filename in filenames:
        if os.path.splitext(filename)[1] == '.png':
          filename_base = '_'.join(filename.split('_')[:-1])
          target_root = os.path.join('gtFine_trainvaltest', 'gtFine', split, os.path.basename(root))
          self.inputs.append(os.path.join(root, filename_base + '_leftImg8bit.png'))
          self.targets.append(os.path.join(target_root, filename_base + '_gtFine_labelIds.png'))

  def __len__(self):
    return len(self.inputs)

  def __getitem__(self, i):
    # Load images and perform augmentations with PIL
    input, target = Image.open(self.inputs[i]), Image.open(self.targets[i])
    # Random uniform crop
    if self.crop is not None:
      w, h = input.size
      x1, y1 = random.randint(0, w - self.crop), random.randint(0, h - self.crop)
      input, target = input.crop((x1, y1, x1 + self.crop, y1 + self.crop)), target.crop((x1, y1, x1 + self.crop, y1 + self.crop))
    # Random horizontal flip
    if self.flip:
      if random.random() < 0.5:
        input, target = input.transpose(Image.FLIP_LEFT_RIGHT), target.transpose(Image.FLIP_LEFT_RIGHT)

    # Convert to tensors
    w, h = input.size
    input = torch.ByteTensor(torch.ByteStorage.from_buffer(input.tobytes())).view(h, w, 3).permute(2, 0, 1).float().div(255)
    target = torch.ByteTensor(torch.ByteStorage.from_buffer(target.tobytes())).view(h, w).long()
    # Normalise input
    input[0].add_(-0.485).div_(0.229)
    input[1].add_(-0.456).div_(0.224)
    input[2].add_(-0.406).div_(0.225)
    # Convert to training labels
    remapped_target = target.clone()
    for k, v in full_to_train.items():
      remapped_target[target == k] = v
    # Create one-hot encoding
    target = torch.zeros(num_classes, h, w)
    for c in range(num_classes):
      target[c][remapped_target == c] = 1
    return input, target, remapped_target  # Return x, y (one-hot), y (index)


train_dataset = CityscapesDataset(split='train', crop=args.crop_size, flip=True)
val_dataset = CityscapesDataset(split='val')
weight_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.workers, pin_memory=True)
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=1, num_workers=args.workers, pin_memory=True)


# Models
# Returns 2D convolutional layer with space-preserving padding
def conv(in_planes, out_planes, kernel_size=3, stride=1, dilation=1, bias=False, transposed=False):
  if transposed:
    layer = nn.ConvTranspose2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=1, output_padding=1, dilation=dilation, bias=bias)
    # Bilinear interpolation init
    w = torch.Tensor(kernel_size, kernel_size)
    centre = kernel_size % 2 == 1 and stride - 1 or stride - 0.5
    for y in range(kernel_size):
      for x in range(kernel_size):
        w[y, x] = (1 - abs((x - centre) / stride)) * (1 - abs((y - centre) / stride))
    layer.weight.data.copy_(w.div(in_planes).repeat(out_planes, in_planes, 1, 1))
  else:
    padding = (kernel_size + 2 * (dilation - 1)) // 2
    layer = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias)
  if bias:
    init.constant(layer.bias, 0)
  return layer


# Returns 2D batch normalisation layer
def bn(planes):
  layer = nn.BatchNorm2d(planes)
  # Use mean 0, standard deviation 1 init
  init.constant(layer.weight, 1)
  init.constant(layer.bias, 0)
  return layer


class FeatureResNet(ResNet):
  def __init__(self):
    super().__init__(BasicBlock, [3, 4, 6, 3], 1000)

  def forward(self, x):
    x1 = self.conv1(x)
    x = self.bn1(x1)
    x = self.relu(x)
    x2 = self.maxpool(x)
    x = self.layer1(x2)
    x3 = self.layer2(x)
    x4 = self.layer3(x3)
    x5 = self.layer4(x4)
    return x1, x2, x3, x4, x5


class SegResNet(nn.Module):
  def __init__(self, num_classes, pretrained_net):
    super().__init__()
    self.pretrained_net = pretrained_net
    self.relu = nn.ReLU(inplace=True)
    self.conv5 = conv(512 + num_classes, 256, stride=2, transposed=True)
    self.bn5 = bn(256)
    self.conv6 = conv(256 + num_classes, 128, stride=2, transposed=True)
    self.bn6 = bn(128)
    self.conv7 = conv(128 + num_classes, 64, stride=2, transposed=True)
    self.bn7 = bn(64)
    self.conv8 = conv(64 + num_classes, 64, stride=2, transposed=True)
    self.bn8 = bn(64)
    self.conv9 = conv(64 + num_classes, 32, stride=2, transposed=True)
    self.bn9 = bn(32)
    self.conv10 = conv(32, num_classes, kernel_size=7, gain='linear')
    init.constant(self.conv10.weight, 0)  # Zero init

  def forward(self, x):
    x1, x2, x3, x4, x5 = self.pretrained_net(x)
    x = self.relu(self.bn5(self.conv5(x5)))
    x = self.relu(self.bn6(self.conv6(x + x4)))
    x = self.relu(self.bn7(self.conv7(x + x3)))
    x = self.relu(self.bn8(self.conv8(x + x2)))
    x = self.relu(self.bn9(self.conv9(x + x1)))
    x = self.conv10(x)
    return x


# Training/Testing
pretrained_net = FeatureResNet()
pretrained_net.load_state_dict(models.resnet34(pretrained=True).state_dict())
net = SegResNet(num_classes, pretrained_net).cuda()
crit = nn.BCELoss().cuda()
optimiser = optim.RMSprop(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
scores, mean_scores = [], []


def train(e):
  net.train()
  for i, (input, target, _) in enumerate(train_loader):
    optimiser.zero_grad()
    input, target = Variable(input.cuda(async=True)), Variable(target.cuda(async=True))
    output = F.sigmoid(net(input))
    loss = crit(output, target)
    print(e, i, loss.data[0])
    loss.backward()
    optimiser.step()


# Calculates class intersections over unions
def iou(pred, target):
  ious = []
  # Ignore IoU for background class
  for cls in range(num_classes - 1):
    pred_inds = pred == cls
    target_inds = target == cls
    intersection = (pred_inds[target_inds]).long().sum().data.cpu()[0]  # Cast to long to prevent overflows
    union = pred_inds.long().sum().data.cpu()[0] + target_inds.long().sum().data.cpu()[0] - intersection
    if union == 0:
      ious.append(float('nan'))  # If there is no ground truth, do not include in evaluation
    else:
      ious.append(intersection / max(union, 1))
  return ious


def test(e):
  net.eval()
  total_ious = []
  for i, (input, _, target) in enumerate(val_loader):
    input, target = Variable(input.cuda(async=True), volatile=True), Variable(target.cuda(async=True), volatile=True)
    output = F.log_softmax(net(input))
    b, _, h, w = output.size()
    pred = output.permute(0, 2, 3, 1).contiguous().view(-1, num_classes).max(1)[1].view(b, h, w)
    total_ious.append(iou(pred, target))

    # Save images
    if i % 25 == 0:
      pred = pred.data.cpu()
      pred_remapped = pred.clone()
      # Convert to full labels
      for k, v in train_to_full.items():
        pred_remapped[pred == k] = v
      # Convert to colour image
      pred = pred_remapped
      pred_colour = torch.zeros(b, 3, h, w)
      for k, v in full_to_colour.items():
        pred_r = torch.zeros(b, 1, h, w)
        pred_r[(pred == k)] = v[0]
        pred_g = torch.zeros(b, 1, h, w)
        pred_g[(pred == k)] = v[1]
        pred_b = torch.zeros(b, 1, h, w)
        pred_b[(pred == k)] = v[2]
        pred_colour.add_(torch.cat((pred_r, pred_g, pred_b), 1))
      save_image(pred_colour[0].float().div(255), os.path.join('results', str(e) + '_' + str(i) + '.png'))

  # Calculate average IoU
  total_ious = torch.Tensor(total_ious).transpose(0, 1)
  ious = torch.Tensor(num_classes - 1)
  for i, class_iou in enumerate(total_ious):
    ious[i] = class_iou[class_iou == class_iou].mean()  # Calculate mean, ignoring NaNs
  print(ious, ious.mean())
  scores.append(ious)

  # Save weights and scores
  torch.save(net.state_dict(), os.path.join('results', str(e) + '_net.pth'))
  torch.save(scores, os.path.join('results', 'scores.pth'))

  # Plot scores
  mean_scores.append(ious.mean())
  es = list(range(len(mean_scores)))
  plt.plot(es, mean_scores, 'b-')
  plt.xlabel('Epoch')
  plt.ylabel('Mean IoU')
  plt.xlim(xmin=0, xmax=args.epochs)
  plt.savefig(os.path.join('results', 'ious.eps'))
  plt.close()


test(0)
for e in range(1, args.epochs + 1):
  train(e)
  test(e)
