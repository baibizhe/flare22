#Toddo: save np.uint8 format.
# save_nii = nib.Nifti1Image(seg_mask.astype(np.uint8), input_nii.affine, input_nii.header)
# nib.save(save_nii, join(save_path, name.split('_0000.nii.gz')[0]+'.nii.gz'))
import argparse
import os
import matplotlib.pyplot as plt
import  numpy as np
import torch
import torchio as tio
from skimage.util import montage
from tqdm import trange
import  nibabel as nib
from kakabaseline import ResUNET
from train import get_model
from utils import CustomImageDataset, CustomValidImageDataset, resizeFun
import SimpleITK as sikt
from tqdm import  tqdm


def find_file(file_path):
    file_list=[]
    if os.path.isfile( file_path):#判断是否为文件，此为基例，递归终止点
        file_list.append(file_path)
    else:           #如果是目录，执行下边的程序
        for file_ls in os.listdir( file_path):#循环目录中的文件
            file_list.extend(find_file(os.path.join( file_path,file_ls)))#再次判断目录中的文件，实现递归，函数调用函数本身
    return file_list #退出最终的文件列表


def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("--dataDirPath1", type=str, default="Validation-20220418T141721Z-001")
    arg("--dataDirPath2", type=str, default="Validation-20220418T141721Z-002")

    arg("--modelPath", type=str, default="ResUnet3D_best.pth")
    args = parser.parse_args()
    print(args)
    # dataDirPath = "Validation-20220418T141721Z-001/Validation"
    # args.dataDirPath = "data/FLARE22_LabeledCase50-20220324T003930Z-001/images"
    # allImages = find_file(args.dataDirPath)

    allImages = find_file(args.dataDirPath1)
    allImages.extend(find_file(args.dataDirPath2))
    assert  len(allImages) ==50
    crop_pad = tio.CropOrPad((128, 512, 512))
    standardize_only_segmentation = tio.ZNormalization(masking_method=tio.ZNormalization.mean)
    resizeTo128 = tio.Resize(target_shape=(128,128,128))

    validTransformForImage = tio.Compose([
        crop_pad,
        standardize_only_segmentation,
    ])
    inferDataset = CustomValidImageDataset(CTImagePath=allImages,
                                      labelPath=None,
                                      labelTransform=None,
                                      imgTransform=None)
    inferDataloader = torch.utils.data.DataLoader(inferDataset, batch_size=1, num_workers=2,
                                             shuffle=False)
    if torch.cuda.is_available():
        print("using GPU 3090!!!!!!!!!!!!!!!!")
        device = torch.device("cuda:%d" % 0)
    else:
        device = "cpu"
        print('==> Using CPU')

    trainedModel = get_model(device)
    trainedModel.load_state_dict(torch.load(args.modelPath),strict=True)
    input_nii = nib.load(allImages[0])
    depth=0
    for idx,image in tqdm(enumerate(inferDataloader)):
        originalshape = image.shape[1:]
        depth+=originalshape[0]
        print(depth)
        # image = validTransformForImage(image)
        # image = resizeTo128(image).unsqueeze(0).to(device)
        # output =trainedModel(image)
        # output = torch.argmax(output, 1).cpu().numpy()
        # finaloutput = resizeFun(output[0],originalshape).astype(np.uint8)
        # # finaloutput = np.transpose(finaloutput,(1,0,2))
        # print("finaloutput {}".format(finaloutput.shape),originalshape)
        # # print(finaloutput.shape)
        # save_nii = nib.Nifti1Image(finaloutput.astype(np.uint8), input_nii.affine, input_nii.header)
        # save_nii.set_data_dtype(np.uint8)
        # nib.save(save_nii, os.path.join("submission","overoverfitting",allImages[idx].split("\\")[-1].split('_0000.nii.gz')[0]+'.nii.gz'))


        # print(allImages[idx].split("\\")[-1].split('_0000.nii.gz')[0]+'.nii.gz'        if idx %20==0:
        # fig, ax1 = plt.subplots(1, 1, figsize=(40, 40), dpi=100)
        # ax1.imshow(montage(finaloutput[20:len(output) - 15]), cmap='bone')
        # fig.show()
        # break
        # print(idx,image.shape)

        # break

    pass
if __name__ == '__main__':
    main()
