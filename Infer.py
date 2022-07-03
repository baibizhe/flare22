#Toddo: save np.uint8 format.
# save_nii = nib.Nifti1Image(seg_mask.astype(np.uint8), input_nii.affine, input_nii.header)
# nib.save(save_nii, join(save_path, name.split('_0000.nii.gz')[0]+'.nii.gz'))
import argparse
import os
import matplotlib.pyplot as plt
import monai
import  numpy as np
import torch
import torchio as tio
import  nibabel as nib
from monai.data import Dataset, DataLoader, decollate_batch
from monai.handlers import from_engine
from monai.inferers import sliding_window_inference

from get_data_loaders import get_data_loaders
from utils import CustomValidImageDataset
from tqdm import  tqdm
from monai.transforms import (
    EnsureChannelFirstd,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    ScaleIntensityRanged,
    EnsureTyped,
    Invertd, RandFlipd, RandShiftIntensityd, ResizeWithPadOrCropd,
)



def find_file(file_path):
    file_list=[]
    if os.path.isfile( file_path):#判断是否为文件，此为基例，递归终止点
        file_list.append(file_path)
    else:           #如果是目录，执行下边的程序
        for file_ls in os.listdir( file_path):#循环目录中的文件
            file_list.extend(find_file(os.path.join( file_path,file_ls)))#再次判断目录中的文件，实现递归，函数调用函数本身
    return file_list #退出最终的文件列表

def get_infer_loaders(args):
    allImages = find_file(args.dataDirPath1)
    allImages.extend(find_file(args.dataDirPath2))
    assert  len(allImages) ==50
    data_dicts = [
        {"image": image_name}
        for image_name in allImages
    ]
    val_org_transforms = Compose(
        [
            LoadImaged(keys=["image" ]),
            EnsureChannelFirstd(keys=["image" ]),
            # ResizeWithPadOrCropd(keys=["image" ], spatial_size=(512, 512, args.patch_size[2])),
            #
            # Orientationd(keys=["image" ], axcodes="RAS"),

            # EnsureTyped(keys=["image" ]),
        ]
    )
    val_org_ds = Dataset(
        data=data_dicts, transform=val_org_transforms)
    post_transforms = Compose([
        EnsureTyped(keys=["image"]),
        Invertd(
            keys=["pred"],
            transform=val_org_transforms,
            orig_keys="image",
            meta_keys=["pred_meta_dict"],
            meta_key_postfix="meta_dict",
            nearest_interp=False,
            to_tensor=True,
        ),
        # AsDiscreted(keys="pred", argmax=True, to_onehot=2),
        # AsDiscreted(keys="label", to_onehot=2),
    ])
    val_org_loader = DataLoader(val_org_ds, batch_size=1,shuffle=False, num_workers=1)

    return post_transforms,  val_org_loader


def get_infer_model(device,args):
    outputChannel = 14

    model = monai.networks.nets.SwinUNETR(img_size=args.patch_size, in_channels=1, out_channels=outputChannel,
                                          feature_size=12,depths=[1,1,1,1]).to(device)
    # model.load_state_dict(torch.load(args.modelPath),strict=True)

    return model


def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("--dataDirPath1", type=str, default="Validation-20220418T141721Z-001")
    arg("--dataDirPath2", type=str, default="Validation-20220418T141721Z-002")

    arg("--modelPath", type=str, default="expOutput\\pretrain+96patch\\pretrain+96patch266.pth")
    arg("--patch-size", type=tuple, default=(64,64,64))

    args = parser.parse_args()
    print(args)
    # dataDirPath = "Validation-20220418T141721Z-001/Validation"
    args.dataDirPath = "data/FLARE22_LabeledCase50-20220324T003930Z-001/images"
    allImages = find_file(args.dataDirPath)

    post_transforms, val_loader = get_infer_loaders(args)


    if torch.cuda.is_available():
        print("using GPU 3090!!!!!!!!!!!!!!!!")
        device = torch.device("cuda:%d" % 0)
    else:
        device = "cpu"
        print('==> Using CPU')

    trainedModel = get_infer_model(device,args)
    input_nii = nib.load(allImages[0])
    depth=0
    trainedModel.eval()
    with torch.no_grad():

        for idx,val_data in tqdm(enumerate(val_loader)):

            val_inputs = val_data["image"].cuda()
            roi_size = args.patch_size
            sw_batch_size = 1
            val_data["pred"] = sliding_window_inference(
                val_inputs, roi_size, sw_batch_size, trainedModel)
            val_data["pred"] = torch.argmax(val_data["pred"], dim=1, keepdim=True)
            print(val_data["pred"].shape)

            val_data = [post_transforms(i) for i in decollate_batch(val_data)]
            val_outputs = from_engine(["pred"])(val_data)
            print(val_outputs[0].shape)



            finaloutput = np.transpose(val_outputs[0][0].cpu().numpy(),(1,0,2))
            save_nii = nib.Nifti1Image(finaloutput.astype(np.uint8), input_nii.affine, input_nii.header)
            save_nii.set_data_dtype(np.uint8)
            nib.save(save_nii, os.path.join("submission","overoverfitting",
                                            os.path.split("\\")[-1].split('_0000.nii.gz')[0] + '.nii.gz'))





    pass
if __name__ == '__main__':
    main()
