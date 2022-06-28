#Toddo: save np.uint8 format.
# save_nii = nib.Nifti1Image(seg_mask.astype(np.uint8), input_nii.affine, input_nii.header)
# nib.save(save_nii, join(save_path, name.split('_0000.nii.gz')[0]+'.nii.gz'))
import argparse
import os
import matplotlib.pyplot as plt
import  numpy as np
import torch
import torchio as tio
import  nibabel as nib
from train import get_model
from utils import CustomValidImageDataset
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
    standardize_only_segmentation = tio.ZNormalization(masking_method=tio.ZNormalization.mean)
    resize = tio.Resize(target_shape=(256, 128, 128))
    rescale = tio.RescaleIntensity((-1, 1))

    validTransformForImage = tio.Compose([
        resize,
        rescale,
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
    for idx,batch in tqdm(enumerate(inferDataloader)):
        image, path = batch[0],batch[1][0]
        originalshape = image.shape
        resizeBack = tio.Resize(target_shape=originalshape[1:])

        image = validTransformForImage(image)

        output = trainedModel(image.unsqueeze(0).cuda())
        output = torch.argmax(output, 1).cpu().numpy()

        finaloutput = resizeBack(output).astype(np.uint8)[0].T
        finaloutput = np.transpose(finaloutput,(1,0,2))
        print("finaloutput {}".format(finaloutput.shape),originalshape)
        print(np.unique(finaloutput))
        save_nii = nib.Nifti1Image(finaloutput.astype(np.uint8), input_nii.affine, input_nii.header)
        save_nii.set_data_dtype(np.uint8)
        nib.save(save_nii, os.path.join("submission","overoverfitting",path.split("\\")[-1].split('_0000.nii.gz')[0]+'.nii.gz'))


        # print(allImages[idx].split("\\")[-1].split('_0000.nii.gz')[0]+'.nii.gz'
        if idx %5==0:
            # fig, ax1 = plt.subplots(1, 1, figsize=(40, 40), dpi=100)
            # ax1.imshow(montage(image.cpu().numpy()[0][20:len(output) - 15]), cmap='bone')
            # ax1.imshow(montage(finaloutput[:,:,20:len(finaloutput) - 15]), cmap='bone')
            plt.subplot(3,1,1)
            plt.imshow(finaloutput[:,:,50])
            plt.subplot(3,1,2)
            plt.imshow(finaloutput[:,:,40])
            plt.subplot(3,1,3)
            plt.imshow(finaloutput[:,:,60])

            plt.show()
            # fig.show()
        # break
        # print(idx,image.shape)

        # break

    pass
if __name__ == '__main__':
    main()
