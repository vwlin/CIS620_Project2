import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
import utils
import random
from densenet import densenet

import learn2learn as l2l
from learn2learn.data.transforms import NWays, KShots, LoadData, RemapLabels, FilterLabels


def pairwise_distances_logits(a, b):
    n = a.shape[0]
    m = b.shape[0]
    logits = -((a.unsqueeze(1).expand(n, m, -1) -
                b.unsqueeze(0).expand(n, m, -1))**2).sum(dim=2)
    return logits


def accuracy(predictions, targets):
    predictions = predictions.argmax(dim=1).view(targets.shape)
    return (predictions == targets).sum().float() / targets.size(0)


class Convnet(nn.Module):

    def __init__(self, x_dim=3, hid_dim=64, z_dim=64):
        super().__init__()
        self.encoder = l2l.vision.models.ConvBase(output_size=z_dim,
                                                  hidden=hid_dim,
                                                  channels=x_dim,
                                                  max_pool=True)
        self.out_channels = 1600

    def forward(self, x):
        x = self.encoder(x)
        return x.view(x.size(0), -1)


def fast_adapt(model, batch, ways, shot, query_num, metric=None, device=None):
    if metric is None:
        metric = pairwise_distances_logits
    if device is None:
        device = model.device()
    data, labels = batch
    data = data.to(device)
    labels = labels.to(device)
    n_items = shot * ways

    # Sort data samples by labels
    # TODO: Can this be replaced by ConsecutiveLabels ?
    sort = torch.sort(labels)
    data = data.squeeze(0)[sort.indices].squeeze(0)
    labels = labels.squeeze(0)[sort.indices].squeeze(0)

    # Compute support and query embeddings
    embeddings = model(data)
    support_indices = np.zeros(data.size(0), dtype=bool)
    selection = np.arange(ways) * (shot + query_num)
    for offset in range(shot):
        support_indices[selection + offset] = True
    query_indices = torch.from_numpy(~support_indices)
    support_indices = torch.from_numpy(support_indices)
    support = embeddings[support_indices]
    support = support.reshape(ways, shot, -1).mean(dim=1)
    query = embeddings[query_indices]
    labels = labels[query_indices].long()

    logits = pairwise_distances_logits(query, support)
    loss = F.cross_entropy(logits, labels)
    acc = accuracy(logits, labels)
    return loss, acc

def fast_adapt_generate_label(model, batch, ways, shot, query_num, metric=None, device=None):
    if metric is None:
        metric = pairwise_distances_logits
    if device is None:
        device = model.device()

    data, labels = batch
    data = data.to(device)
    labels = labels.to(device)
    n_items = shot * ways

    # Sort data samples by labels
    # TODO: Can this be replaced by ConsecutiveLabels ?
    sort = torch.sort(labels)
    data = data.squeeze(0)[sort.indices].squeeze(0)
    labels = labels.squeeze(0)[sort.indices].squeeze(0)

    # Compute support and query embeddings
    embeddings = model(data)
    support_indices = np.zeros(data.size(0), dtype=bool)
    selection = np.arange(ways) * (shot + query_num)
    for offset in range(shot):
        support_indices[selection + offset] = True
    query_indices = torch.from_numpy(~support_indices)
    support_indices = torch.from_numpy(support_indices)
    support = embeddings[support_indices]
    support = support.reshape(ways, shot, -1).mean(dim=1)
    query = embeddings[query_indices]
    labels = labels[query_indices].long()

    logits = pairwise_distances_logits(query, support)
    pseudo_labels = logits.argmax(dim=1).view(labels.shape)
    return data,pseudo_labels

def fast_adapt_with_pseudo_label(model, data,labels, ways, shot, query_num, metric=None, device=None):
    if metric is None:
        metric = pairwise_distances_logits
    if device is None:
        device = model.device()
    n_items = shot * ways

    # Compute support and query embeddings
    embeddings = model(data)
    support_indices = np.zeros(data.size(0), dtype=bool)
    selection = np.arange(ways) * (shot + query_num)
    for offset in range(shot):
        support_indices[selection + offset] = True
    query_indices = torch.from_numpy(~support_indices)
    support_indices = torch.from_numpy(support_indices)
    support = embeddings[support_indices]
    support = support.reshape(ways, shot, -1).mean(dim=1)
    query = embeddings[query_indices]

    logits = pairwise_distances_logits(query, support)
    loss = F.cross_entropy(logits, labels)
    acc = accuracy(logits, labels)
    return loss, acc


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='Convnet')
    parser.add_argument('--max-epoch', type=int, default=250) # previously, 250
    parser.add_argument('--train-way', type=int, default=5) # 30
    parser.add_argument('--shot', type=int, default=1)
    parser.add_argument('--train-query', type=int, default=1) # 15

    parser.add_argument('--test-way', type=int, default=5)
    parser.add_argument('--test-shot', type=int, default=1)
    parser.add_argument('--test-query', type=int, default=1) # 1

    parser.add_argument('--gpu', default=0)
    args = parser.parse_args()
    print(args)

    device = torch.device('cpu')
    if args.gpu and torch.cuda.device_count():
        print("Using gpu")
        torch.cuda.manual_seed(43)
        device = torch.device('cuda')

    
    # Create model
    if 'Convnet' in args.model:
        model = Convnet()
    elif 'vgg' in args.model:
        model = torchvision.models.vgg11(pretrained=True)
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Identity() # gives us penultimate layer (a.k.a., embeddings)
        print(model)
    elif 'densenet' in args.model:
        model = densenet(
                num_classes=14,
                depth=40,
                growthRate=12,
                compressionRate=2,
                dropRate=0,
            )
        model.fc = nn.Identity() # add this!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        print(model)
    elif 'resnet18' in args.model:
        model = torchvision.models.resnet18(pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = nn.Identity()
        print(model)
    elif 'resnet50' in args.model:
        model = torchvision.models.resnet50(pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = nn.Identity()
        print(model)

    model.to(device)

    # path_data = '~/data'
    # train_dataset = l2l.vision.datasets.MiniImagenet(
    #     root=path_data, mode='train')
    # valid_dataset = l2l.vision.datasets.MiniImagenet(
    #     root=path_data, mode='validation')
    # test_dataset = l2l.vision.datasets.MiniImagenet(
    #     root=path_data, mode='test')

    # Hello, CURE_TSR_OG
    data_transforms = transforms.Compose([transforms.Resize([32, 32]), transforms.ToTensor()])#, utils.l2normalize, utils.standardization])

    lvl0_train_dir = './CURE_TSR_OG/Real_Train/ChallengeFree/'
    lvl5_test_dir = './CURE_TSR_OG/Real_Train/Snow-5/'
    curetsr_lvl0 = utils.CURETSRDataset(lvl0_train_dir, data_transforms)
    curetsr_lvl5 = utils.CURETSRDataset(lvl5_test_dir, data_transforms)

    # lvl0_train_dir = './CURE_TSR_Yahan_Shortcut/Real_Train/ChallengeFree/'
    # lvl5_test_dir = './CURE_TSR_Yahan_Shortcut/Real_Train/Snow-5/'
    # curetsr_lvl0 = datasets.ImageFolder(lvl0_train_dir, transform=data_transforms)
    # print("first image, label is ", curetsr_lvl0[0])
    # curetsr_lvl5 = datasets.ImageFolder(lvl5_test_dir, transform=data_transforms)

    meta_curetsr_lvl0 = l2l.data.MetaDataset(curetsr_lvl0)
    meta_curetsr_lvl5 = l2l.data.MetaDataset(curetsr_lvl5)

    train_dataset = meta_curetsr_lvl0
    valid_dataset = meta_curetsr_lvl0
    test_dataset = meta_curetsr_lvl5

    classes = list(range(14)) # 14 classes of stop signs
    random.shuffle(classes)
    # Changes, end!

    train_dataset = l2l.data.MetaDataset(train_dataset)
    train_transforms = [
        FilterLabels(train_dataset, classes[:8]),
        NWays(train_dataset, args.train_way),
        KShots(train_dataset, args.train_query + args.shot),
        LoadData(train_dataset),
        RemapLabels(train_dataset),
    ]
    train_tasks = l2l.data.TaskDataset(train_dataset, task_transforms=train_transforms)
    train_loader = DataLoader(train_tasks, pin_memory=True, shuffle=True)

    valid_dataset = l2l.data.MetaDataset(valid_dataset)
    valid_transforms = [
        FilterLabels(valid_dataset, classes[8:14]),
        NWays(valid_dataset, args.test_way),
        KShots(valid_dataset, args.test_query + args.test_shot),
        LoadData(valid_dataset),
        RemapLabels(valid_dataset),
    ]
    valid_tasks = l2l.data.TaskDataset(valid_dataset,
                                       task_transforms=valid_transforms,
                                       num_tasks=200)
    valid_loader = DataLoader(valid_tasks, pin_memory=True, shuffle=True)

    test_dataset = l2l.data.MetaDataset(test_dataset)
    test_transforms = [
        FilterLabels(test_dataset, classes[8:14]),
        NWays(test_dataset, args.test_way),
        KShots(test_dataset, args.test_query + args.test_shot),
        LoadData(test_dataset),
        RemapLabels(test_dataset),
    ]
    test_tasks = l2l.data.TaskDataset(test_dataset,
                                      task_transforms=test_transforms,
                                      num_tasks=2000)
    test_loader = DataLoader(test_tasks, pin_memory=True, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=20, gamma=0.5)

    # meta-train on level 0 (source domain)
    print("meta-training on level 0\n")
    for epoch in range(1, args.max_epoch + 1):
        model.train()

        loss_ctr = 0
        n_loss = 0
        n_acc = 0

        for i in range(100):
            batch = next(iter(train_loader))
            loss, acc = fast_adapt(model,
                                   batch,
                                   args.train_way,
                                   args.shot,
                                   args.train_query,
                                   metric=pairwise_distances_logits,
                                   device=device)

            loss_ctr += 1
            n_loss += loss.item()
            n_acc += acc

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        lr_scheduler.step()

        print('epoch {}, train, loss={:.4f} acc={:.4f}'.format(
            epoch, n_loss/loss_ctr, n_acc/loss_ctr))

        model.eval()

        loss_ctr = 0
        n_loss = 0
        n_acc = 0
        for i, batch in enumerate(valid_loader):
            loss, acc = fast_adapt(model,
                                   batch,
                                   args.test_way,
                                   args.test_shot,
                                   args.test_query,
                                   metric=pairwise_distances_logits,
                                   device=device)

            loss_ctr += 1
            n_loss += loss.item()
            n_acc += acc

        print('epoch {}, val, loss={:.4f} acc={:.4f}'.format(
            epoch, n_loss/loss_ctr, n_acc/loss_ctr))

    # get pseudo-labels for level 4 and meta-train level 0 model on level 4 with pseudo-labels
    print('\n\nmeta-training level 0 model on level 4 data and pseudolabels')
    inter_dir = './CURE_TSR_OG/Real_Train/Snow-4/'
    curetsr_lvl4 = utils.CURETSRDataset(inter_dir, data_transforms)

    meta_curetsr_lvl4 = l2l.data.MetaDataset(curetsr_lvl4)
    train_level_dataset = meta_curetsr_lvl4
    valid_level_dataset = meta_curetsr_lvl4

    train_level_dataset = l2l.data.MetaDataset(train_level_dataset)
    train_level_transforms = [
        FilterLabels(train_level_dataset, classes[:8]),
        NWays(train_level_dataset, args.train_way),
        KShots(train_level_dataset, args.train_query + args.shot),
        LoadData(train_level_dataset),
        RemapLabels(train_level_dataset),
    ]
    train_level_tasks = l2l.data.TaskDataset(train_level_dataset, task_transforms=train_level_transforms)
    train_level_loader = DataLoader(train_level_tasks, pin_memory=True, shuffle=True)

    valid_level_dataset = l2l.data.MetaDataset(valid_level_dataset)
    valid_level_transforms = [
        FilterLabels(valid_level_dataset, classes[8:14]),
        NWays(valid_level_dataset, args.test_way),
        KShots(valid_level_dataset, args.test_query + args.test_shot),
        LoadData(valid_level_dataset),
        RemapLabels(valid_level_dataset),
    ]
    valid_level_tasks = l2l.data.TaskDataset(valid_level_dataset,
                                    task_transforms=valid_level_transforms,
                                    num_tasks=200)
    valid_level_loader = DataLoader(valid_level_tasks, pin_memory=True, shuffle=True)

    optimizer_l = torch.optim.Adam(model.parameters(), lr=0.001)
    lr_scheduler_l = torch.optim.lr_scheduler.StepLR(
    optimizer_l, step_size=20, gamma=0.5)
    
    for epoch in range(1, args.max_epoch + 1): 
        loss_ctr = 0
        n_loss = 0
        n_acc = 0

        for i in range(100):
            batch = next(iter(train_level_loader))
            model.eval()
            data, pseudo_labels = fast_adapt_generate_label(model,
                                batch,
                                args.train_way,
                                args.shot,
                                args.train_query,
                                metric=pairwise_distances_logits,
                                device=device)
            
            model.train()
            loss, acc = fast_adapt_with_pseudo_label(model, data, pseudo_labels, 
                            args.train_way,
                            args.shot,
                            args.train_query, metric=pairwise_distances_logits,
                            device=device)
            loss_ctr += 1
            n_loss += loss.item()
            n_acc += acc

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        #lr_scheduler_l.step()

        print('epoch {}, train, loss={:.4f} acc={:.4f}'.format(
            epoch, n_loss/loss_ctr, n_acc/loss_ctr))

        model.eval()

        loss_ctr = 0
        n_loss = 0
        n_acc = 0
        for i, batch in enumerate(valid_level_loader):
            loss, acc = fast_adapt(model,
                                batch,
                                args.test_way,
                                args.test_shot,
                                args.test_query,
                                metric=pairwise_distances_logits,
                                device=device)

            loss_ctr += 1
            n_loss += loss.item()
            n_acc += acc

        print('epoch {}, val, loss={:.4f} acc={:.4f}'.format(
            epoch, n_loss/loss_ctr, n_acc/loss_ctr))
    
    loss_ctr = 0
    n_acc = 0
    
    # meta-test on level 5 (target domain)
    print("\n\nmeta-testing on level 5\n\n")
    for i, batch in enumerate(test_loader, 1):
        loss, acc = fast_adapt(model,
                               batch,
                               args.test_way,
                               args.test_shot,
                               args.test_query,
                               metric=pairwise_distances_logits,
                               device=device)
        loss_ctr += 1
        n_acc += acc
        print('batch {}: {:.2f}({:.2f})'.format(
            i, n_acc/loss_ctr * 100, acc * 100))