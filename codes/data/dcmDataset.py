import numpy as np
import torch.utils.data as data
import SimpleITK as sitk
import pydicom
import utils.util as util
import random
import os, glob
import torch

class dcmDataset(data.Dataset):
    def __init__(self, opt):
        super(dcmDataset, self).__init__()
        self.FILL_RATIO_THRESHOLD = 0.8
        self.opt = opt
        self.in_folder = opt['dataroot_LR']
        self.tar_folder = opt['dataroot_HR']
        self.mask_folder = opt['maskroot_HR']

        # 3d voxel size
        if self.opt['phase'] == 'train':
            self.ps = (opt['LR_slice_size'], opt['LR_size'], opt['LR_size'])

        self.uids = opt['uids']
        if opt['subset'] is not None:
           self.uids = self.uids[:opt['subset']]

        self.scale = opt['scale']
        self.ToTensor = util.ImgToTensor()

    def __getitem__(self, index):
        uid = self.uids[index]
        # LR
        uid_path = os.path.join(self.in_folder, uid)
        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(uid_path)
        reader.SetFileNames(dicom_names)
        image = reader.Execute()
        spacings = list(image.GetSpacing())
        vol_in = sitk.GetArrayFromImage(image) # D W H
        vol_in = np.clip(vol_in, -1000, 500) + 1000 # range [0 1500]
        vol_in = vol_in[::-1,:,:] # itk reads depth backwards, fixed it
        
        if self.tar_folder:
           # implement read HR 
           uid_path = os.path.join(self.tar_folder, uid)
           reader = sitk.ImageSeriesReader()
           dicom_names = reader.GetGDCMSeriesFileNames(uid_path)
           reader.SetFileNames(dicom_names)
           image = reader.Execute()
           spacings = list(image.GetSpacing())
           vol_tar = sitk.GetArrayFromImage(image) # D W H
           vol_tar = np.clip(vol_tar, -1000, 500) + 1000 # range [0 1500]
           vol_tar = vol_tar[::-1,:,:] # itk reads depth backwards, fixed it

        # vol_tar = None
        vol_in = np.expand_dims(vol_in, axis=0) # [B C D W H]
        vol_tar = np.expand_dims(vol_tar, axis=0)
        vol_in, vol_tar = self.ToTensor(vol_in), self.ToTensor(vol_tar)

        out_dict = {'LR': vol_in, 'HR': vol_tar, 'uid': uid, 'spacings': spacings}
        return out_dict

    def __len__(self):
        return len(self.uids)
