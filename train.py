import argparse
import os
import random

import matplotlib.pyplot as plt
import monai.networks.nets
import numpy as np
import pytorch_toolbelt.losses
import torch
import torchio as tio
from monai.inferers import sliding_window_inference
from skimage.util import montage
from timm.utils import AverageMeter
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange
from torch.optim import  AdamW
from get_data_loaders import get_data_loaders
from utils import CustomImageDataset, compute_dice_coefficient, CustomValidImageDataset
from models.unet3d.model import  ResidualUNet3D
from monai.data import CacheDataset, DataLoader, Dataset, decollate_batch
from monai.handlers.utils import from_engine

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
    scaler = GradScaler()
    dice_coefficients = AverageMeter()

    for batch_idx, batch in enumerate(train_loader):
        data, target = batch["image"].to(device), batch["label"].to(device).long()
        optimizer.zero_grad()
        # print(data.shape,target.shape)
        with autocast():
            output = model(data)

            loss = loss_fn(output.unsqueeze(2), target)  # * 1000
            output = torch.argmax(output, 1)

        dice_coefficients.update(compute_dice_coefficient(output.detach().cpu().numpy(), target.cpu().numpy()), 1)
        losses.update(loss.item(), data.size(0))
        torch.nn.utils.clip_grad_norm_(model.parameters(),100)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.step()
    return losses.avg, dice_coefficients.avg


def val_epoch_metrics(model, val_org_loader,lossFun, device, post_transforms ):
    model.eval()
    losses = AverageMeter()
    dice_coefficients = AverageMeter()
    output = None

    with torch.no_grad():
        for idx,val_data in enumerate(val_org_loader):
            val_inputs = val_data["image"].to(device)
            roi_size = (128, 128, 128)
            sw_batch_size = 1
            val_data["pred"] = sliding_window_inference(
                val_inputs, roi_size, sw_batch_size, model)
            val_data["pred"]=torch.argmax(val_data["pred"],dim=1,keepdim=True)
            val_data = [post_transforms(i) for i in decollate_batch(val_data)]
            val_outputs, val_labels = from_engine(["pred", "label"])(val_data)
            dice_coefficients.update(compute_dice_coefficient(val_outputs[0].numpy().astype(int), val_labels[0].numpy().astype(int)), 1)
            if idx==1:
                plt.figure("check", (12, 6))
                plt.subplot(1, 2, 1)
                plt.title("image")
                plt.imshow(val_outputs[0].cpu()[0].numpy()[:, :, 60], cmap="gray")
                plt.subplot(1, 2, 2)
                plt.title("label")
                plt.imshow(val_labels[0].cpu()[0].numpy()[:, :, 60])
                plt.show()
            print(val_outputs[0].numpy().shape, val_labels[0].numpy().shape)
            # compute metric for current iteration

        # aggregate the final mean dice result
        # reset the status for next validation round

    return  100,dice_coefficients.avg

def val_epoch(model, train_loader, optimizer, device, epoch, trainepochs, loss_fn,validTransformForImage,validTransformForLabel):
    model.eval()
    losses = AverageMeter()
    dice_coefficients = AverageMeter()
    output = None

    for batch_idx, (data, target) in enumerate(train_loader):
        originalShape = target.shape
        upsampler = tio.Resize(originalShape[1:])

        resizeData = validTransformForImage(data).unsqueeze(0).to(device)
        resizeTarget = validTransformForLabel(target).long().to(device)
        with torch.no_grad():
            output = model(resizeData)

            loss = loss_fn(output, resizeTarget.unsqueeze(0))
            output = torch.argmax(output, 1)

            losses.update(loss.item(), data.size(0))
            output = upsampler(output.cpu().numpy())
            dice_coefficients.update(compute_dice_coefficient(output.astype(int), target.numpy()), 1)

        if batch_idx % 5 == 0:
            if epoch % 5 == 0:
                print("保存图像")
                target = resizeTarget.cpu().numpy()[0, :]
                output = output[0, :]

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
    # model = UNet(in_dim=1, out_dim=outputChannel, num_filters=4) #baseline的UNET3D
    # model = ResUNET(outputChannel=outputChannel, feature_scale=8) # 自己实现的resunet3D ， 这个可能是最稳定的版本

    # model = ResidualUNet3D(in_channels=1,out_channels=outputChannel,final_sigmoid=False,num_levels=3,f_maps=16,layer_order='cbl')
    model = monai.networks.nets.SwinUNETR(in_channels=1,out_channels=outputChannel,feature_size=12,img_size=(128,128,128),depths=(1, 1, 1, 1),
                                          use_checkpoint=True,num_heads=(2,4,8,16))


    # 这个是git上面一个resunet3D ， 花样很多

    # model = ViTVNet.ViTVNet(img_size=(256, 128, 128))  #队友推荐的VIT3D
    model.to(device)
    return model


def get_transform(target_shape):
    # crop_pad = tio.CropOrPad((128, 512, 512))
    # resize = tio.Resize(target_shape=(256, 256, 256))
    resize = tio.Resize(target_shape=target_shape)

    standardize_only_segmentation = tio.ZNormalization(masking_method=tio.ZNormalization.mean)
    rescale = tio.RescaleIntensity((-1, 1))

    random_anisotropy = tio.RandomAnisotropy(p=0.5)
    """
    Researchers typically use anisotropic resampling for preprocessing before feeding the images into a neural network. We can simulate this effect downsampling our image along a specific dimension and resampling back to an isotropic spacing. Of course, this is a lossy operation, but that's the point! We want to increase the diversity of our dataset so that our models generalize better. Images in the real world have different artifacts and are not always isotropic, so this is a great transform for medical images.
    """


    random_flip = tio.RandomFlip(axes=['Right'], flip_probability=0.5)
    trainTransform = tio.Compose([
        resize,
        random_flip
        # random_flip,
    ])
    trainTransformWillChangeValue = tio.Compose([
        # standardize_only_segmentation,
        random_anisotropy,
        rescale,
    ])
    validTransformForImage = tio.Compose([
        resize,
        rescale,
        # standardize_only_segmentation,
    ])
    validTransformForLaebl = tio.Compose([
        resize,
    ])
    return trainTransform, trainTransformWillChangeValue, validTransformForImage, validTransformForLaebl

def main():
    # seed_everything()
    # torch.autograd.set_detect_anomaly(True)
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("--batch-size", type=int, default=1

        )
    arg("--epochs", type=int, default=1000)
    arg("--lr", type=float, default=0.000005)
    arg("--workers", type=int, default=6)
    arg("--model", type=str, default="ResUnet3D")
    arg("--target-shape",type=int,default=(256,256,256))
    #     arg("--test_mode", type=str2bool, default="false",choices=[True,False])
    arg("--optimizer", type=str, default="AdamW")
    arg("--taskname", type=str, default="Super+Resunet+256")

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
    data_dicts = [
        {"image": image_name, "label": label_name}
        for image_name, label_name in zip(imgPaths, labelPath)
    ]
    train_files, val_files = data_dicts[:-9], data_dicts[-9:]
    # check_ds = Dataset(data=val_files, transform=val_transforms)
    splitIndex = int(len(imgPaths) * 0.9)

    writer = SummaryWriter(os.path.join(baseRoot, "logs"))
    #
    # trainTransform, \
    # trainTransformWillChangeValue, \
    # validTransformForImage, \
    # validTransformForLaebl = get_transform(args.target_shape)
    #
    # trainDataset = CustomImageDataset(CTImagePath=imgPaths[0:splitIndex],
    #                                   labelPath=labelPath[0:splitIndex],
    #                                   labelTransform=trainTransform,
    #                                   TransformWillChangeValue=trainTransformWillChangeValue,
    #                                   imgTransform=trainTransform)
    # valDataset = CustomValidImageDataset(CTImagePath=imgPaths[splitIndex:],
    #                                      labelPath=labelPath[splitIndex:],
    #                                      imgTransform=None,
    #                                      labelTransform=None,
    #                                      )
    #
    # print("total images:", len(trainDataset))
    # train_loader = torch.utils.data.DataLoader(trainDataset, batch_size=args.batch_size, num_workers=args.workers,
    #                                            shuffle=True, prefetch_factor=4, pin_memory=True)
    # val_loader = torch.utils.data.DataLoader(valDataset, batch_size=1, num_workers=args.workers,
    #                                          shuffle=False, pin_memory=True)

    train_loader,val_loader,post_transforms = get_data_loaders(None)
    model = get_model(device=device)
    if args.resumePath != '':
        print("loading model from {}".format(args.resumePath))
        model.load_state_dict(torch.load(args.resumePath), strict=True)
    else:
        print("new training")
        # model._init_weights()
    optimizer = eval(args.optimizer)(model.parameters(), lr=args.lr, weight_decay=1e-5)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.95, last_epoch=-1)
    warmup_epochs = 3
    T_mult = 2
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=warmup_epochs,
                                                                     T_mult=T_mult)
    # will restart at 5+5*2=15，15+10*2=35，35+20 * 2=75

    epochs = args.epochs
    valid_dice = 0
    lossFun = torch.nn.CrossEntropyLoss()
    # lossFun = pytorch_toolbelt.losses.DiceLoss(mode="multiclass") # CE 的loss很容易训飞了
    trainLosses = AverageMeter()
    trainDiceCoefficients = AverageMeter()
    validLosses = AverageMeter()
    validDiceCoefficients = AverageMeter()
    validLossEpoch, validDiceEpoch=0,0
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
            # trainLossEpoch, trainDiceEpoch = 0,0

            if epoch % 2 == 0:
                validLossEpoch, validDiceEpoch = val_epoch_metrics(model=model,val_org_loader=val_loader,lossFun=lossFun,device="cuda",post_transforms=post_transforms)

            trainLosses.update(trainLossEpoch)
            trainDiceCoefficients.update(trainDiceEpoch)
            validLosses.update(validLossEpoch)
            validDiceCoefficients.update(validDiceEpoch)
            lr = optimizer.param_groups[0]["lr"]

            t.set_postfix({"Train loss ":trainLossEpoch,
                           "Train Dice resized": trainDiceEpoch,
                           "Valid loss ": validLossEpoch,
                           "Valid full dice ": validDiceEpoch,
                           "lr": lr,

                           })

            if validDiceEpoch > valid_dice:
                torch.save(model.state_dict(), os.path.join(baseRoot,args.taskname +str(epoch)+ ".pth"))
                print("模型保存"
                      "")
                valid_dice = validDiceEpoch
            writer.add_scalars(
                "loss", {"train loss": trainLossEpoch, "val loss": validLossEpoch}, epoch + 1,
            )
            writer.add_scalars(
                "LR", {"lr": lr, }, epoch + 1,
            )
            writer.add_scalars(
                "Dice",
                {"Train Dice resized": trainDiceEpoch, "Valid dice full size": validDiceEpoch},
                epoch + 1,
            )
            scheduler.step()


if __name__ == "__main__":
    main()
