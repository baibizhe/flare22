import argparse
import os
import random
from types import SimpleNamespace

import matplotlib.pyplot as plt
import monai.networks.nets
import numpy as np
import torch
import wandb
from monai.inferers import sliding_window_inference
from timm.utils import AverageMeter
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange
from get_data_loaders import get_data_loaders
from models.ssl_head import SSLHead
from tensor_board import Tensorboard
from utils import compute_dice_coefficient, init_weights
from monai.data import decollate_batch
from monai.handlers.utils import from_engine
from torch.optim import  AdamW

def seed_everything(seed=2022):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def train_epoch(model, train_loader, optimizer, device, epoch, trainepochs, loss_fn1, loss_fn2):
    model.train()
    losses = AverageMeter()
    scaler = GradScaler()
    dice_coefficients = AverageMeter()

    for batch_idx, batch in enumerate(train_loader):
        data, target = batch["image"].to(device), batch["label"].to(device).long()
        optimizer.zero_grad()
        with autocast():
            output = model(data)

            loss = loss_fn1(output.unsqueeze(2), target)  # * 1000
            loss += loss_fn2(output, target) * 0.2
            output = torch.argmax(output, 1)

        dice_coefficients.update(compute_dice_coefficient(output.detach().cpu().numpy(), target.cpu().numpy()), len(batch))
        losses.update(loss.item(), data.size(0))
        torch.nn.utils.clip_grad_norm_(model.parameters(), 100)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.step()
    return losses.avg, dice_coefficients.avg


def val_epoch_metrics(model, val_org_loader, post_transforms, patchshape):
    model.eval()
    dice_coefficients = AverageMeter()
    out_up , label_up =None,None
    with torch.no_grad():
        for idx, val_data in enumerate(val_org_loader):
            val_inputs = val_data["image"].cuda()
            roi_size = patchshape
            sw_batch_size = 1
            val_data["pred"] = sliding_window_inference(
                val_inputs, roi_size, sw_batch_size, model)
            val_data["pred"] = torch.argmax(val_data["pred"], dim=1, keepdim=True)
            val_data = [post_transforms(i) for i in decollate_batch(val_data)]
            val_outputs, val_labels = from_engine(["pred", "label"])(val_data)
            dice_coefficients.update(
                compute_dice_coefficient(val_outputs[0].numpy().astype(int), val_labels[0].numpy().astype(int)), 1)
            if idx == 1:
                out_up = val_outputs[0].cpu()[0].numpy()[:, :, 60]
                label_up = val_labels[0].cpu()[0].numpy()[:, :, 60]


    return  dice_coefficients.avg,out_up,label_up


def get_model(device, config):
    outputChannel = 14

    model = monai.networks.nets.SwinUNETR(img_size=config.patchshape, in_channels=1, out_channels=outputChannel,
                                          feature_size=48).cuda()
    init_weights(model, init_type="kaiming")
    if len(config.pretrain_path) > 2:
        args = SimpleNamespace(use_checkpoint=True, roi_x=config.patchshape[0], roi_y=config.patchshape[1],
                               roi_z=config.patchshape[2], sw_batch_size=1, feature_size=48, in_channels=1,
                               spatial_dims=3, dropout_path_rate=0)
        pretrain = SSLHead(args, upsample='deconv')
        state_dict = torch.load(config.pretrain_path, map_location=torch.device('cpu'))["state_dict"]
        pretrain = torch.nn.DataParallel(pretrain)
        pretrain.load_state_dict(state_dict, strict=True)
        pretrain = pretrain.module
        model.swinViT = pretrain.swin_vit
    if config.resumePath != '':
        print("loading model from {}".format(config.resumePath))
        model.load_state_dict(torch.load(config.resumePath), strict=True)
    else:
        print("new training")
        # model._init_weights()
    model.to(device)
    return model


def main():
    # seed_everything()
    # torch.autograd.set_detect_anomaly(True)
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("--batch-size", type=int, default=1

        )
    arg("--epochs", type=int, default=1000)
    arg("--lr", type=float, default=0.0005)
    arg("--workers", type=int, default=6)
    arg("--model", type=str, default="ResUnet3D")
    arg("--patchshape", type=tuple, default=(96, 96, 96))
    #     arg("--test_mode", type=str2bool, default="false",choices=[True,False])
    arg("--optimizer", type=str, default="AdamW")
    arg("--taskname", type=str, default="pretrain+629night+96patch+baseaug")

    arg("--pretrain-path", type=str, default='model_bestValRMSE.pt')
    arg("--resumePath", type=str, default='')
    arg("--use-wandb", action="store_true",)

    arg(
        "--device-ids",
        type=str,
        default="0",
        help="For example 0,1 to run on two GPUs",
    )
    args = parser.parse_args()
    print(args)
    os.environ['WANDB_API_KEY'] = "55a895793519c48a6e64054c9b396629d3e41d10"

    if args.use_wandb:
        mywandb = Tensorboard(config=args)
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
    post_transforms, train_loader, val_loader = get_loaders(args)
    model = get_model(device=device, config=args)

    optimizer = eval(args.optimizer)(model.parameters(), lr=args.lr, weight_decay=1e-5)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.95, last_epoch=-1)
    warmup_epochs = 3
    T_mult = 2
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=warmup_epochs,
                                                                     T_mult=T_mult)
    # will restart at 5+5*2=15，15+10*2=35，35+20 * 2=75

    epochs = args.epochs
    CELossFun = torch.nn.CrossEntropyLoss()
    DiceLossFun = monai.losses.DiceCELoss(to_onehot_y=True,
                                          softmax=True)
    trainLosses,trainDiceCoefficients,validLosses,validDiceCoefficients = AverageMeter(),AverageMeter(),AverageMeter(),AverageMeter()
    validLossEpoch, validDiceEpoch ,valid_dice = 0, 0 , 0
    for param in model.swinViT.parameters():
        param.requires_grad = False
    with trange(epochs) as t:
        for epoch in t:
            if epoch == 3:
                for param in model.swinViT.parameters():
                    param.requires_grad = True
            t.set_description('Epoch %i' % epoch)
            trainLossEpoch, trainDiceEpoch = train_epoch(model=model,
                                                         train_loader=train_loader,
                                                         optimizer=optimizer,
                                                         device=device,
                                                         epoch=epoch,
                                                         loss_fn1=CELossFun,
                                                         loss_fn2=DiceLossFun,
                                                         trainepochs=epochs)
            # trainLossEpoch, trainDiceEpoch = 0,0

            if epoch % 2 == 0:
                validDiceEpoch,out_up,label_up  = val_epoch_metrics(model=model,
                                                                   val_org_loader=val_loader,
                                                                   post_transforms=post_transforms,
                                                                   patchshape=args.patchshape)

            trainLosses.update(trainLossEpoch)
            trainDiceCoefficients.update(trainDiceEpoch)
            validDiceCoefficients.update(validDiceEpoch)
            lr = optimizer.param_groups[0]["lr"]


            info_dict = {"Train loss ": trainLossEpoch,
                           "Train Dice resized": trainDiceEpoch,
                           "Valid full dice ": validDiceEpoch,
                           "lr": lr,

                           }
            if args.use_wandb:
                mywandb.upload_wandb_info(info_dict=info_dict)
                plt.figure("check", (12, 6))
                plt.subplot(1, 2, 1)
                plt.title("image")
                plt.imshow(out_up)
                plt.subplot(1, 2, 2)
                plt.title("label")
                plt.imshow(label_up)

                mywandb.tensor_board.log({"data": wandb.Image(plt)})
            else:
                plt.figure("check", (12, 6))
                plt.subplot(1, 2, 1)
                plt.title("image")
                plt.imshow(out_up)
                plt.subplot(1, 2, 2)
                plt.title("label")
                plt.imshow(label_up)

                plt.savefig(os.path.join("outPutImages",'output{}.png'.format(epoch)))
            t.set_postfix(info_dict)

            if validDiceEpoch > valid_dice:
                torch.save(model.state_dict(), os.path.join(baseRoot, args.taskname + str(epoch) + ".pth"))
                print("模型保存")
                valid_dice = validDiceEpoch

            scheduler.step()


def get_loaders(args):
    dataDirPath = "data/FLARE22_LabeledCase50-20220324T003930Z-001"
    imgPaths = list(
        map(lambda x: os.path.join(dataDirPath, "images", x), os.listdir(os.path.join(dataDirPath, "images"))))
    labelPath = list(
        map(lambda x: os.path.join(dataDirPath, "labels", x), os.listdir(os.path.join(dataDirPath, "labels"))))
    train_loader, val_loader, post_transforms = get_data_loaders(args)
    return post_transforms, train_loader, val_loader


if __name__ == "__main__":
    main()
