import torch
from torch.utils.data import DataLoader
from torchmetrics.classification import MulticlassMatthewsCorrCoef
from shapenet_dataset import ShapenetDataset
from point_net import PointNetClassHead
import torch.optim as optim
from point_net_loss import PointNetLoss
from train_loop import Trainer
import numpy as np
from utils import CATEGORIES

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

ROOT = 'shapenetcore_partanno_segmentation_benchmark_v0'

EPOCHS = 100
LR = 0.001
REG_WEIGHT = 0.001
NUM_CLASSES = 50  
NUM_TRAIN_POINTS = 2500
NUM_TEST_POINTS = 10000
NUM_CLASSES = 16

GLOBAL_FEATS = 1024
BATCH_SIZE = 32

alpha = np.ones(NUM_CLASSES)
alpha[0] = 0.5
alpha[4] = 0.5
alpha[-1] = 0.5
gamma = 2


train_dataset = ShapenetDataset(ROOT, npoints=NUM_TRAIN_POINTS, split='train', classification=True, normalize=False)
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)

valid_dataset = ShapenetDataset(ROOT, npoints=NUM_TRAIN_POINTS, split='valid', classification=True, normalize=False)
valid_dataloader = DataLoader(valid_dataset, batch_size=32, shuffle=True, num_workers=4)

test_dataset = ShapenetDataset(ROOT, npoints=NUM_TEST_POINTS, split='test', classification=True, normalize=False)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True, num_workers=4)




classifier = PointNetClassHead(k=NUM_CLASSES, num_global_feats=GLOBAL_FEATS).to(device)  
optimizer = optim.Adam(classifier.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.0001, max_lr=0.01, step_size_up=2000, cycle_momentum=False)
criterion = PointNetLoss(alpha=alpha, gamma=gamma, reg_weight=REG_WEIGHT).to(device)
mcc_metric = MulticlassMatthewsCorrCoef(num_classes=NUM_CLASSES).to(device)

training_loop = Trainer(classifier, train_dataloader, valid_dataloader, optimizer, scheduler, criterion, device, 32, EPOCHS, mcc_metric)
training_loop.train()
