import argparse
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchio as tio
from skimage.util import montage
from timm.utils import AverageMeter
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange
from torch.optim import  AdamW
from Vit import ViTVNet
from utils import CustomImageDataset, compute_dice_coefficient, CustomValidImageDataset


def seed_everything(seed=2022):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def train_epoch(model, train_loader, optimizer, device, epoch, trainepochs, loss_fn):
    model.train()
    losses = AverageMeter()
    #     loss_fn = torch.nn.MSELoss()
    scaler = GradScaler()
    dice_coefficients = AverageMeter()

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device).long()
        optimizer.zero_grad()

        with autocast():
            output = model(data)
            loss = loss_fn(output, target)  # * 1000
            output = torch.argmax(output, 1)
        dice_coefficients.update(compute_dice_coefficient(output.detach().cpu().numpy(), target.cpu().numpy()), 1)
        losses.update(loss.item(), data.size(0))
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.step()
    return losses.avg, dice_coefficients.avg

    # if epoch %20 == 0:
    #     np.save(file="100output.npy",arr=model(data).detach().cpu().numpy())
    #     np.save(file="100data.npy",arr=data.cpu().numpy())
    #     np.save(file="100target.npy",arr=target.cpu().numpy())


def val_epoch(model, train_loader, optimizer, device, epoch, trainepochs, loss_fn):
    model.eval()
    losses = AverageMeter()
    dice_coefficients = AverageMeter()
    output = None
    target_shape = (128, 128, 128)
    target_shape = (256, 256, 256)

    resizeTo128 = tio.Resize(target_shape=target_shape)
    for batch_idx, (data, target) in enumerate(train_loader):
        batchSize = data.shape[0]
        resizeData = torch.zeros(size=(batchSize, 1, 256, 256, 256))
        targetResized = torch.zeros(size=(batchSize, 1, 256, 256, 256))
        for i in range(batchSize):
            resizeData[i][0] = resizeTo128(data[i])
            targetResized[i][0] = resizeTo128(target[i].unsqueeze(0))
        resizeData = resizeData.to(device)
        data, targetResized = data.to(device), targetResized.squeeze(1).long().to(device)
        with torch.no_grad():
            output = model(resizeData)
            loss = loss_fn(output, targetResized)
            losses.update(loss.item(), data.size(0))
            output = torch.argmax(output, 1)
            # outputOriginal = resizeFun(output.cpu().numpy(),(batchSize,256,512,512))
            dice_coefficients.update(compute_dice_coefficient(output.cpu().numpy().astype(int), target.numpy()), 1)

        if batch_idx % 5 == 0:
            if epoch % 5 == 0:
                print("保存图像")
                target = targetResized.cpu().numpy()[0, :]
                output = output.cpu().numpy()[0, :]

                fig, ax1 = plt.subplots(1, 1, figsize=(40, 40), dpi=100)
                ax1.imshow(montage(output[20:len(output) - 15]), cmap='bone')
                fig.savefig('outPutImages/output_epoch{}.png'.format(epoch))
                fig, ax1 = plt.subplots(1, 1, figsize=(40, 40), dpi=100)
                ax1.imshow(montage(target[20:len(target) - 15]), cmap='bone')
                fig.savefig('outPutImages/target_epoch{}.png'.format(epoch))
                plt.close('all')
    return losses.avg, dice_coefficients.avg


def get_model(device):
    outputChannel = 14
    # model = UNet(in_dim=1, out_dim=outputChannel, num_filters=4)

    # model = ResUNET(outputChannel=outputChannel, feature_scale=4)
    model = ViTVNet.ViTVNet(img_size=(256, 256, 256))
    model.to(device)
    return model


def get_transform():
    # crop_pad = tio.CropOrPad((128, 512, 512))
    resize = tio.Resize(target_shape=(256, 256, 256))
    standardize_only_segmentation = tio.ZNormalization(masking_method=tio.ZNormalization.mean)
    random_anisotropy = tio.RandomAnisotropy(p=0.5)
    """
    Researchers typically use anisotropic resampling for preprocessing before feeding the images into a neural network. We can simulate this effect downsampling our image along a specific dimension and resampling back to an isotropic spacing. Of course, this is a lossy operation, but that's the point! We want to increase the diversity of our dataset so that our models generalize better. Images in the real world have different artifacts and are not always isotropic, so this is a great transform for medical images.
    """
    random_flip = tio.RandomFlip(axes=['inferior-superior'], flip_probability=0.5)
    trainTransform = tio.Compose([
        resize,
        random_flip,
    ])
    trainTransformWillChangeValue = tio.Compose([
        standardize_only_segmentation,
    ])

    validTransformForImage = tio.Compose([
        resize,
        standardize_only_segmentation,

    ])

    validTransformForLaebl = tio.Compose([
        resize,
    ])

    return trainTransform, trainTransformWillChangeValue, validTransformForImage, validTransformForLaebl


def main():
    # seed_everything()
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("--batch-size", type=int, default=2)
    arg("--epochs", type=int, default=500)
    arg("--lr", type=float, default=0.01)
    arg("--workers", type=int, default=6)
    arg("--model", type=str, default="ResUnet3D")
    #     arg("--test_mode", type=str2bool, default="false",choices=[True,False])
    arg("--optimizer", type=str, default="AdamW")
    arg("--taskname", type=str, default="Supervised+VIT256+Binit")

    arg("--resumePath", type=str, default='')
    arg(
        "--device-ids",
        type=str,
        default="0",
        help="For example 0,1 to run on two GPUs",
    )
    args = parser.parse_args()
    print(args)
    if not os.path.exists("outPutImages"):
        try:
            os.makedirs("outPutImages")
        except:
            pass
    baseRoot = os.path.join("expOutput", args.taskname)
    if not os.path.exists(baseRoot):
        try:
            os.makedirs(baseRoot)
        except:
            pass
    device = torch.device("cuda:%d" % 0)
    dataDirPath = "data/FLARE22_LabeledCase50-20220324T003930Z-001"
    # dataDirPath = "data/FLARE22_LabeledCase50"

    imgPaths = list(
        map(lambda x: os.path.join(dataDirPath, "images", x), os.listdir(os.path.join(dataDirPath, "images"))))
    labelPath = list(
        map(lambda x: os.path.join(dataDirPath, "labels", x), os.listdir(os.path.join(dataDirPath, "labels"))))
    # print(imgPaths)
    # print(labelPath)
    splitIndex = int(len(imgPaths) * 0.8)

    writer = SummaryWriter(os.path.join(baseRoot, "logs"))

    trainTransform, \
    trainTransformWillChangeValue, \
    validTransformForImage, \
    validTransformForLaebl = get_transform()

    trainDataset = CustomImageDataset(CTImagePath=imgPaths[0:splitIndex],
                                      labelPath=labelPath[0:splitIndex],
                                      labelTransform=trainTransform,
                                      TransformWillChangeValue=trainTransformWillChangeValue,
                                      imgTransform=trainTransform)
    valDataset = CustomValidImageDataset(CTImagePath=imgPaths[splitIndex:],
                                         labelPath=labelPath[splitIndex:],
                                         imgTransform=validTransformForImage,
                                         labelTransform=validTransformForLaebl,

                                         )

    print("total images:", len(trainDataset))
    train_loader = torch.utils.data.DataLoader(trainDataset, batch_size=args.batch_size, num_workers=args.workers,
                                               shuffle=True, prefetch_factor=4, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(valDataset, batch_size=1, num_workers=args.workers,
                                             shuffle=False, pin_memory=True)
    model = get_model(device=device)
    if args.resumePath != '':
        print("loading model from {}".format(args.resumePath))
        model.load_state_dict(torch.load(args.resumePath), strict=True)
    else:
        print("new training")
    optimizer = eval(args.optimizer)(model.parameters(), lr=3e-4, weight_decay=1e-5)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.95, last_epoch=-1)
    warmup_epochs = 5
    T_mult = 2
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=warmup_epochs,
                                                                     T_mult=T_mult)
    # will restart at 5+5*2=15，15+10*2=35，35+20 * 2=75

    epochs = args.epochs
    valid_dice = 0
    lossFun = torch.nn.CrossEntropyLoss()
    trainLosses = AverageMeter()
    trainDiceCoefficients = AverageMeter()
    validLosses = AverageMeter()
    validDiceCoefficients = AverageMeter()
    with trange(epochs) as t:
        for epoch in t:
            t.set_description('Epoch %i' % epoch)
            trainLossEpoch, trainDiceEpoch = train_epoch(model=model,
                                                         train_loader=train_loader,
                                                         optimizer=optimizer,
                                                         device=device,
                                                         epoch=epoch,
                                                         loss_fn=lossFun,
                                                         trainepochs=epochs)

            if epoch % 5 == 0:
                validLossEpoch, validDiceEpoch = val_epoch(model, val_loader, optimizer, device, epoch, epochs, lossFun)
            # validLossEpoch, validDiceEpoch = 1,1
            trainLosses.update(trainLossEpoch)
            trainDiceCoefficients.update(trainDiceEpoch)
            validLosses.update(validLossEpoch)
            validDiceCoefficients.update(validDiceEpoch)
            lr = optimizer.param_groups[0]["lr"]

            t.set_postfix({"Train loss avg": trainLosses.avg,
                           "Train Dice resized": trainDiceCoefficients.avg,
                           "Valid loss avg": validLosses.avg,
                           "Valid dice full size": validDiceCoefficients.avg,
                           "lr": lr,

                           })

            if validDiceEpoch > valid_dice:
                torch.save(model.state_dict(), os.path.join(baseRoot,args.taskname + ".pth"))
                valid_dice = validDiceEpoch
            writer.add_scalars(
                "loss", {"train loss": trainLosses.avg, "val loss": validLosses.avg}, epoch + 1,
            )
            writer.add_scalars(
                "LR", {"lr": lr, }, epoch + 1,
            )
            writer.add_scalars(
                "Dice",
                {"Train Dice resized": trainDiceCoefficients.avg, "Valid dice full size": validDiceCoefficients.avg},
                epoch + 1,
            )
            scheduler.step()


if __name__ == "__main__":
    main()
