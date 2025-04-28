from collections import Counter
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split
import torchvision.models as models
from torch.optim import lr_scheduler
import torch.optim as optim
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
import os
import numpy as np
import matplotlib.pyplot as plt

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(256),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


dataset = datasets.ImageFolder(root='simpsons_dataset_v2/train/', transform=preprocess)

# Собираем все метки из dataset.samples
labels = [label for _, label in dataset.samples]

# Считаем, сколько раз каждая метка встречается
counts = Counter(labels)

# Выводим по именам классов
for idx, cnt in counts.most_common():
    class_name = dataset.classes[idx]
    print(f"{class_name}: {cnt}")
