"""
    [1] https://github.com/olivesgatech/CURE-TSR/blob/master/train.py (lines 58-65)
"""

import torch
from torchvision.transforms import transforms
import learn2learn as l2l
# custom imports
import utils


# Data loading code
traindir = './CAML-Pytorch/CURE_TSR/Real_Train/ChallengeFree/'
testdir = './CAML-Pytorch/CURE_TSR/Real_Test/ChallengeFree/'

#BATCH_SIZE = 64
#WORKERS = 3
train_dataset = utils.CURETSRDataset(traindir, transforms.Compose([
    transforms.Resize([28, 28]), transforms.ToTensor(), utils.l2normalize, utils.standardization]))
#train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=WORKERS, pin_memory=True)
test_dataset = utils.CURETSRDataset(testdir, transforms.Compose([
    transforms.Resize([28, 28]), transforms.ToTensor(), utils.l2normalize, utils.standardization]))
#test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=WORKERS, pin_memory=True)

# l2l custom dataset wrapper
dataset = l2l.data.MetaDataset(train_dataset)
transforms = [  # Easy to define your own transform
    l2l.data.transforms.NWays(dataset, n=5),
    l2l.data.transforms.KShots(dataset, k=4),
    l2l.data.transforms.LoadData(dataset),
]
taskset = l2l.data.TaskDataset(dataset, transforms, num_tasks=20)
for task in taskset:
    X, y = task
    print(X.shape, y.shape)