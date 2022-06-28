import os
from monai.utils import first, set_determinism
from monai.transforms import (
    AsDiscrete,
    AsDiscreted,
    EnsureChannelFirstd,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandCropByPosNegLabeld,
    SaveImaged,
    ScaleIntensityRanged,
    Spacingd,
    EnsureTyped,
    EnsureType,
    Invertd,
)
from monai.transforms.croppad.dictionary import ResizeWithPadOrCropd
# from monai.handlers.utils import from_engine
from monai.networks.nets import UNet
from monai.networks.layers import Norm
from monai.metrics import DiceMetric
from monai.losses import DiceLoss
from monai.inferers import sliding_window_inference
from monai.data import CacheDataset, DataLoader, Dataset, decollate_batch
from monai.config import print_config
from monai.apps import download_and_extract
import torch
import matplotlib.pyplot as plt
import tempfile
import shutil
import os
import glob
def get_data_loaders(config):
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
    splitIndex = int(len(imgPaths) * 0.9)
    train_files, val_files = data_dicts[:splitIndex], data_dicts[splitIndex:]
    train_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            # Orientationd(keys=["image", "label"], axcodes="RAS"),
            # # Spacingd(keys=["image", "label"], pixdim=(
            # #     1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
            ResizeWithPadOrCropd(keys=["image", "label"],spatial_size=(512,512,128)),
            ScaleIntensityRanged(
                keys=["image"], a_min=-57, a_max=164,
                b_min=0.0, b_max=1.0, clip=True,
            ),
            CropForegroundd(keys=["image", "label"], source_key="image"),
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=(128, 128, 128),
                pos=1,
                neg=1,
                num_samples=4,
                image_key="image",
                image_threshold=0,
            ),
            # user can also add other random transforms
            # RandAffined(
            #     keys=['image', 'label'],
            #     mode=('bilinear', 'nearest'),
            #     prob=1.0, spatial_size=(96, 96, 96),
            #     rotate_range=(0, 0, np.pi/15),
            #     scale_range=(0.1, 0.1, 0.1)),
            EnsureTyped(keys=["image", "label"]),
        ]
    )
    val_org_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            # ResizeWithPadOrCropd(keys=["image", "label"], spatial_size=(512, 512, 128)),

            Orientationd(keys=["image"], axcodes="RAS"),
            # Spacingd(keys=["image"], pixdim=(
            #     1.5, 1.5, 2.0), mode="bilinear"),
            ScaleIntensityRanged(
                keys=["image"], a_min=-57, a_max=164,
                b_min=0.0, b_max=1.0, clip=True,
            ),
            # CropForegroundd(keys=["image"], source_key="image"),
            EnsureTyped(keys=["image", "label"]),
        ]
    )

    train_ds = Dataset(
        data=train_files, transform=train_transforms,
         )
    # train_ds = Dataset(data=train_files, transform=train_transforms)

    # use batch_size=2 to load images and use RandCropByPosNegLabeld
    # to generate 2 x 4 images for network training
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=1)

    val_org_ds = Dataset(
        data=val_files, transform=val_org_transforms)
    check_ds = Dataset(data=val_files, transform=val_org_transforms)
    check_loader = DataLoader(check_ds, batch_size=1)
    check_data = first(check_loader)
    image, label = (check_data["image"][0][0], check_data["label"][0][0])
    print(f"image shape: {image.shape}, label shape: {label.shape}")
    # plot the slice [:, :, 80]
    plt.figure("check", (12, 6))
    plt.subplot(1, 2, 1)
    plt.title("image")
    plt.imshow(image[:, :, 80], cmap="gray")
    plt.subplot(1, 2, 2)
    plt.title("label")
    plt.imshow(label[:, :, 80])
    plt.show()
    post_transforms = Compose([
        EnsureTyped(keys="pred"),
        Invertd(
            keys="pred",
            transform=val_org_transforms,
            orig_keys="image",
            meta_keys="pred_meta_dict",
            orig_meta_keys="image_meta_dict",
            meta_key_postfix="meta_dict",
            nearest_interp=False,
            to_tensor=True,
        ),
        # AsDiscreted(keys="pred", argmax=True, to_onehot=2),
        # AsDiscreted(keys="label", to_onehot=2),
    ])
    val_org_loader = DataLoader(val_org_ds, batch_size=1,shuffle=False, num_workers=1)
    # val_ds = Dataset(data=val_files, transform=val_transforms)
    return   train_loader,val_org_loader,post_transforms
    # for i in train_loader:
    #     print(i["image"].shape)
    # for j in val_loader:
        # print(j.shape)
if __name__ == '__main__':
    get_data_loaders(None)

