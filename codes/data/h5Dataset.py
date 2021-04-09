import numpy as np
import h5py
import torch.utils.data as data
import utils.util as util
import random
import os
import torch

class h5Dataset(data.Dataset):

    def __init__(self, opt):
        super(h5Dataset, self).__init__()
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
        # body mask
        # first we look at if the random voxel contain 80% of the body mask
        vol_mask = None
        if self.mask_folder:
            with h5py.File(os.path.join(self.mask_folder, uid+'.h5'), 'r') as file:
                IMG_THICKNESS, IMG_WIDTH, IMG_HEIGHT = file['data'].shape
                # random index of the 3D voxel for each patient
                if self.opt['phase'] == 'train':
                    t, w, h = self.ps
                    # randomly search the voxel until 80% filled with ones
                    fill_ratio = 0.
                    while fill_ratio < self.FILL_RATIO_THRESHOLD:
                        rnd_t_HR = random.randint(0, IMG_THICKNESS - int(t * self.scale))
                        rnd_h = random.randint(0, IMG_HEIGHT - h)
                        rnd_w = random.randint(0, IMG_WIDTH - w)
                        fill_ratio = file['data'][rnd_t_HR:rnd_t_HR+int(t*self.scale), rnd_h:rnd_h+h, rnd_w:rnd_w+w].sum() \
                                    / (self.scale * t * w * h)
                    # print(fill_ratio)
                    vol_mask = None
                else:
                # return the whole volume in inference mode
                    vol_mask = file['data'][()]
        # LR
        with h5py.File(os.path.join(self.in_folder, uid+'.h5'), 'r') as file:
            if self.opt['phase'] == 'train':
                vol_in = file['data'][round(rnd_t_HR/self.scale):round(rnd_t_HR/self.scale)+t,
                                      rnd_h:rnd_h+h, rnd_w:rnd_w+w]
            else:
                vol_in = file['data'][()]
        # HR
        vol_tar = None
        if self.tar_folder:
            with h5py.File(os.path.join(self.tar_folder, uid+'.h5'), 'r') as file:
                if self.opt['phase'] == 'train':
                    vol_tar = file['data'][rnd_t_HR:rnd_t_HR+int(t*self.scale), rnd_h:rnd_h+h, rnd_w:rnd_w+w]
                else:
                    vol_tar = file['data'][()]

        vol_in = np.expand_dims(vol_in, axis=0)
        vol_tar = np.expand_dims(vol_tar, axis=0)
        vol_mask = np.expand_dims(vol_mask, axis=0)
        vol_in, vol_tar = self.ToTensor(vol_in), self.ToTensor(vol_tar)
        vol_mask = self.ToTensor(vol_mask, raw_data_range = 1)

        # read LR spacings
        spacings = [] 
        if self.opt['phase'] == 'val':
            config_path = os.path.join(self.in_folder, uid + '.json')
            meta_data = util.read_config(config_path)
            spacings = meta_data['Spacing']
        out_dict = {'LR': vol_in, 'HR': vol_tar, 'mask': vol_mask, 
                    'spacings': spacings, 'uid': uid}

        return out_dict

    def __len__(self):
        return len(self.uids)
