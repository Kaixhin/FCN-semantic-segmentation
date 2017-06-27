from argparse import ArgumentParser
import os
import random
from matplotlib import pyplot as plt
import torch
from torch import optim
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import models
from torchvision.utils import save_image

from data import CityscapesDataset, num_classes, full_to_colour, train_to_full
from model import FeatureResNet, SegResNet


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


# Data
train_dataset = CityscapesDataset(split='train', crop=args.crop_size, flip=True)
val_dataset = CityscapesDataset(split='val')
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=1, num_workers=args.workers, pin_memory=True)


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
  plt.savefig(os.path.join('results', 'ious.png'))
  plt.close()


test(0)
for e in range(1, args.epochs + 1):
  train(e)
  test(e)
