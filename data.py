import os
import random
from PIL import Image
import torch
from torch.utils.data import Dataset


# Labels: -1 license plate, 0 unlabeled, 1 ego vehicle, 2 rectification border, 3 out of roi, 4 static, 5 dynamic, 6 ground, 7 road, 8 sidewalk, 9 parking, 10 rail track, 11 building, 12 wall, 13 fence, 14 guard rail, 15 bridge, 16 tunnel, 17 pole, 18 polegroup, 19 traffic light, 20 traffic sign, 21 vegetation, 22 terrain, 23 sky, 24 person, 25 rider, 26 car, 27 truck, 28 bus, 29 caravan, 30 trailer, 31 train, 32 motorcycle, 33 bicycle
num_classes = 20
full_to_train = {-1: 19, 0: 19, 1: 19, 2: 19, 3: 19, 4: 19, 5: 19, 6: 19, 7: 0, 8: 1, 9: 19, 10: 19, 11: 2, 12: 3, 13: 4, 14: 19, 15: 19, 16: 19, 17: 5, 18: 19, 19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14, 28: 15, 29: 19, 30: 19, 31: 16, 32: 17, 33: 18}
train_to_full = {0: 7, 1: 8, 2: 11, 3: 12, 4: 13, 5: 17, 6: 19, 7: 20, 8: 21, 9: 22, 10: 23, 11: 24, 12: 25, 13: 26, 14: 27, 15: 28, 16: 31, 17: 32, 18: 33, 19: 0}
full_to_colour = {0: (0, 0, 0), 7: (128, 64, 128), 8: (244, 35, 232), 11: (70, 70, 70), 12: (102, 102, 156), 13: (190, 153, 153), 17: (153, 153, 153), 19: (250, 170, 30), 20: (220, 220, 0), 21: (107, 142, 35), 22: (152, 251, 152), 23: (70, 130, 180), 24: (220, 20, 60), 25: (255, 0, 0), 26: (0, 0, 142), 27: (0, 0, 70), 28: (0, 60,100), 31: (0, 80, 100), 32: (0, 0, 230), 33: (119, 11, 32)}


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
