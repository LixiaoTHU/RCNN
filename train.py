#!/usr/bin/env python

from torchvision.datasets import CIFAR10
from torch.nn import DataParallel
from torch.optim.lr_scheduler import MultiStepLR
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import torch
import csv, sys, os, time, argparse

from rcnn import RCNN

os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"

def test(model, testloader, criterion):

    model.eval()
    correct, total = 0, 0
    loss, counter = 0, 0

    with torch.no_grad():
        for (images, labels) in testloader:

            images = images.cuda()
            labels = labels.cuda()
            bs, ncrops, c, h, w = images.size()

            outputs = model(images.view(-1, c, h, w))
            result_avg = outputs.view(bs, ncrops, -1).mean(1)
            _, predicted = torch.max(result_avg.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            loss += criterion(result_avg, labels).item()
            counter += 1

    return loss / counter, correct / total

def load_data(datadir, batch_size, GPU_COUNT):
    transform_train = transforms.Compose([
        transforms.RandomCrop(24),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    transform_test = transforms.Compose([
        transforms.TenCrop(24),
        transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
        transforms.Lambda(lambda crops: torch.stack([transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))(crop) for crop in crops]))])

    trainset = CIFAR10(datadir, transform = transform_train, train = True, download=True)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size = batch_size * GPU_COUNT, shuffle = True, num_workers = 32)

    testset = CIFAR10(datadir, transform = transform_test, train = False, download=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size = batch_size * GPU_COUNT, shuffle = False, num_workers = 32)
    return trainloader, testloader



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train RCNN')
    parser.add_argument('-n', dest='K', type=int, default = 96, help='the parameter K for RCNN')
    parser.add_argument('-b', dest='batch_size', type=int, default=64, help='the batch size in just one gpu, * GPU_COUNT')
    parser.add_argument('-e', dest='epoch', type=int, default=200, help='the training epoch')
    parser.add_argument('-s', dest='save_dir', type=str, default="log.csv", help='the model parameters to be saved')
    parser.add_argument('-l', dest='training_log', type=str, default="weights.pkl", help='the logs to be saved')
    args = parser.parse_args()

    # write header
    with open(args.save_dir, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(["iteration", "train_loss", "val_loss", "acc", "val_acc"])

    # build model and optimizer
    model = RCNN(3, 10, args.K)
    model.cuda()
    model = nn.DataParallel(model)
    # model.load_state_dict(torch.load('160_weights_noagu.pkl'))
    GPU_COUNT = torch.cuda.device_count()
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(model.parameters(), lr = 1e-1, weight_decay = 1e-4, momentum = 0.9, nesterov = True)
    epoch = args.epoch
    scheduler = MultiStepLR(optimizer, milestones=[int(epoch/2),int(epoch*3/4),int(epoch*7/8)], gamma=0.1)

    trainloader, testloader = load_data("~/Datasets", args.batch_size, GPU_COUNT)


    # train
    i = 0
    correct, total = 0, 0
    train_loss, counter = 0, 0

    for epoch in range(0, args.epoch):
        scheduler.step()
        start_time = time.time()
        # iteration over all train data
        for data in trainloader:
            # shift to train mode
            model.train()

            # get the inputs
            inputs, labels = data
            inputs = inputs.cuda()
            labels = labels.cuda()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # count acc,loss on trainset
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            train_loss += loss.item()
            counter += 1

            if i % 200 == 0:
                # get acc,loss on trainset
                acc = correct / total
                train_loss /= counter

                # test
                val_loss, val_acc = test(model, testloader, criterion)
                print('iteration %d , epoch %d:  loss: %.4f  val_loss: %.4f  acc: %.4f  val_acc: %.4f'
                      %(i, epoch, train_loss, val_loss, acc, val_acc))

                # save logs and weights
                with open(args.training_log, 'a') as f:
                    writer = csv.writer(f)
                    writer.writerow([i, train_loss, val_loss, acc, val_acc])
                torch.save(model.state_dict(), args.save_dir)

                # reset counters
                correct, total = 0, 0
                train_loss, counter = 0, 0

            i += 1
        print("current time cost: ", time.time() - start_time)
