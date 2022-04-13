import argparse
import os

import pytorch_toolbelt.losses
import torch
from timm.utils import AverageMeter
from torch.optim import AdamW

from utils import CustomImageDataset, resizeFun
from UnetBaseline import  UNet
from torch.cuda.amp import autocast, GradScaler
import  numpy as np
def train_epoch(model, train_loader, optimizer, device, epoch, trainepochs,loss_fn):
    model.train()
    losses = AverageMeter()
    #     loss_fn = torch.nn.MSELoss()
    scaler = GradScaler()

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device).long()
        optimizer.zero_grad()
        with autocast():
            output = model(data)
            loss = loss_fn(output, target)  # * 1000
        losses.update(loss.item(), data.size(0))
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        optimizer.step()

        if batch_idx % 5 == 0:
            log = ('Train Epoch:[{}/{}({:.0f}%)]\t'
                   'It:[{}/{}({:2.0f}%)]\t'
                   'Loss: {:.4f}({:.4f})'.format(
                epoch, trainepochs, 1. * epoch / trainepochs * 100,
                batch_idx, len(train_loader), 1. * batch_idx / len(train_loader) * 100,
                loss.item(), losses.avg))
            print(log)

def val_epoch(model, train_loader, optimizer, device, epoch, trainepochs,loss_fn):
    model.eval()
    losses = AverageMeter()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        #         optimizer.zero_grad()
        with torch.no_grad():
            output = model(data)
            loss = loss_fn(output, target)
            losses.update(loss.item(), data.size(0))
        #         loss.backward()
        #         optimizer.step()

        if batch_idx % 5 == 0:
            log = ('VAL Epoch:[{}/{}({:.0f}%)]\t'
                   'It:[{}/{}({:2.0f}%)]\t'
                   'Loss: {:.4f}({:.4f})'.format(
                epoch, trainepochs, 1. * epoch / trainepochs * 100,
                batch_idx, len(train_loader), 1. * batch_idx / len(train_loader) * 100,
                loss.item(), losses.avg))
            print(log)
            np.save(file="100output.npy",arr=model(data).detach().cpu().numpy())
            np.save(file="100data.npy",arr=data.cpu().numpy())
            np.save(file="100target.npy",arr=target.cpu().numpy())

    print(losses.avg)
    return losses.avg
def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("--batch-size", type=int, default=2)
    arg("--n-epochs", type=int, default=100)
    arg("--lr", type=float, default=0.0001)
    arg("--workers", type=int, default=12)
    arg("--model", type=str, default="UNet16")
    #     arg("--test_mode", type=str2bool, default="false",choices=[True,False])
    arg("--early_stopping", type=int, default=15)

    arg("--train_class", type=int, default=1, choices=[1, 2, 3, 4])
    arg("--optimizer", type=str, default="Adam")
    arg(
        "--device-ids",
        type=str,
        default="0",
        help="For example 0,1 to run on two GPUs",
    )
    args = parser.parse_args()

    device = torch.device("cuda:%d" % 0)
    dataDirPath = "data/FLARE22_LabeledCase50-20220324T003930Z-001"
    imgPaths = list(
        map(lambda x: os.path.join(dataDirPath, "images", x), os.listdir(os.path.join(dataDirPath, "images"))))
    labelPath = list(
        map(lambda x: os.path.join(dataDirPath, "labels", x), os.listdir(os.path.join(dataDirPath, "labels"))))
    print(imgPaths)
    print(labelPath)
    splitIndex = int(len(imgPaths) * 0.8)
    trainDataset = CustomImageDataset(CTImagePath=imgPaths[0:splitIndex],
                                      labelPath=labelPath[0:splitIndex],
                                      labelTransform=resizeFun,
                                      imgTransform=resizeFun)
    valDataset = CustomImageDataset(CTImagePath=imgPaths[0:splitIndex],
                                      labelPath=labelPath[0:splitIndex],
                                      labelTransform=resizeFun,
                                      imgTransform=resizeFun)

    print("total images:", len(trainDataset))
    train_loader = torch.utils.data.DataLoader(trainDataset, batch_size=args.batch_size, num_workers=args.workers,
                                               shuffle=True)
    val_loader = torch.utils.data.DataLoader(valDataset, batch_size=args.batch_size, num_workers=args.workers,
                                             shuffle=False)
    model = UNet(in_dim=1, out_dim=14, num_filters=4)
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=150, gamma=0.25, last_epoch=-1)

    epochs = 500
    valid_mse = 0
    lossFun = torch.nn.CrossEntropyLoss()
    for epoch in range(1, epochs + 1):
        train_epoch(model=model,
                    train_loader=train_loader,
                    optimizer=optimizer,
                    device=device,
                    epoch=epoch,
                    loss_fn=lossFun,
                    trainepochs=epochs)
        # cur_valid_mse = val_epoch(model, val_loader, optimizer, device, epoch, epochs)
        # if cur_valid_mse > valid_mse:
        #     torch.save(model.state_dict(), "best_4.pth")
        scheduler.step()


if __name__ == "__main__":
    main()