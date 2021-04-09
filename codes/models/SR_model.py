import logging
from collections import OrderedDict
import torch
import torch.nn as nn
import models.lr_scheduler as lr_scheduler
import models.networks as networks
from .base_model import BaseModel
from apex import amp
import apex

logger = logging.getLogger('base')

class SRModel(BaseModel):
    def __init__(self, opt):
        super(SRModel, self).__init__(opt)
        train_opt = opt['train']

        if self.is_train:
            self.netG.train()

            # loss
            loss_type = train_opt['pixel_criterion']
            if loss_type == 'l1':
                self.cri_pix = nn.L1Loss().to(self.device)
            elif loss_type == 'l2':
                self.cri_pix = nn.MSELoss().to(self.device)
            else:
                raise NotImplementedError('Loss type [{:s}] is not recognized.'.format(loss_type))
            self.l_pix_w = train_opt['pixel_weight']

            # optimizers
            wd_G = train_opt['weight_decay_G'] if train_opt['weight_decay_G'] else 0
            optim_params = []
            for k, v in self.netG.named_parameters():  # can optimize for a part of the model
                if v.requires_grad:
                    optim_params.append(v)
                else:
                    logger.warning('Params [{:s}] will not optimize.'.format(k))

            if opt["network_G"]["which_model_G"] == "fsrcnn":
                self.optimizer_G = torch.optim.Adam([
                        {'params': self.netG.first_part.parameters()},
                        {'params': self.netG.mid_part.parameters()},
                        {'params': self.netG.last_part.parameters(), 'lr': train_opt['lr_G'] * 0.1}
                        ], lr=train_opt['lr_G'])
            else:
                self.optimizer_G = torch.optim.Adam(
                    optim_params, lr=train_opt['lr_G'], weight_decay=wd_G)
            self.optimizers.append(self.optimizer_G)

            # schedulers
            if train_opt['lr_scheme'] == 'MultiStepLR':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.MultiStepLR_Restart(optimizer, train_opt['lr_steps'],
                                                         restarts=train_opt['restarts'],
                                                         weights=train_opt['restart_weights'],
                                                         gamma=train_opt['lr_gamma'],
                                                         clear_state=train_opt['clear_state']))
            elif train_opt['lr_scheme'] == 'CosineAnnealingLR':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.CosineAnnealingLR_Restart(optimizer, train_opt['T_period'],
                                                        eta_min=train_opt['eta_min'],
                                                        restarts=train_opt['restarts'],
                                                        weights=train_opt['restart_weights']))
            else:
                raise NotImplementedError('Choose MultiStepLR or CosineAnnealingLR')

            self.log_dict = OrderedDict()
        

    def initialize_amp(self):
        self.netG, self.optimizer_G = amp.initialize(self.netG, self.optimizer_G, opt_level=self.opt['opt_level'])
        if self.opt['gpu_ids']:
            assert torch.cuda.is_available()
            self.netG = nn.DataParallel(self.netG)
            # self.netG = apex.parallel.DistributedDataParallel(self.netG)

    def test(self, data):
        self.netG_eval.eval()

        if self.opt['precision'] == 'fp16':
            var_L_eval = self.var_L.half()
        else:
            var_L_eval = self.var_L

        pt = self.var_L.size(2)
        H = self.var_L.size(3)
        W = self.var_L.size(4)
        # LR slice patch x 2 = HR slice patch
        pt = int(pt * self.opt['scale'])
        # HR slice num
        num_HR = int (data['LR'].size(2) * self.opt['scale'])
        # HR slice overlap
        HR_ot = int(self.opt['scale'] * self.ot)
        self.fake_H = torch.empty(1, 1,  num_HR, H, W, device=self.device)
        if self.opt['precision'] == 'fp16':
            fake_H_in_chunks = torch.empty(self.nt, 1,  pt, H, W, dtype=torch.half, device=self.device)
        else:
            fake_H_in_chunks = torch.empty(self.nt, 1,  pt, H, W, device=self.device)
        # mask, record 1 when there is value in this pixel
        stitch_mask = torch.zeros_like(self.fake_H, device=self.device)
        with torch.no_grad():
            if self.opt['datasets']['val']['full_volume']:
                self.fake_H = self.netG_eval(var_L_eval)
            else :
                for i in range(0, self.nt):
                    fake_H_in_chunks[[i],...] = self.netG_eval(var_L_eval[[i],...])

                # n-1 volumes
                for i in range(0, self.nt - 1):
                    ts, te = i * (pt - HR_ot), i * (pt - HR_ot) + pt
                    # important!
                    # both stitch_mask and fake_H are fp32 for better numerical stability
                    # fp16 inference results are casted into fp32
                    self.fake_H[0, 0, ts:te, :, :] = \
                    (self.fake_H[0, 0, ts:te, :, :] * stitch_mask[0, 0, ts:te, :, :] +
                    fake_H_in_chunks[i,...].float() * (2 - stitch_mask[0, 0, ts:te, :, :])) / 2
                    stitch_mask[0, 0, ts:te, :, :] = 1.
                    # next_block = avg ([1 0] *  next_block +  [1 2] * new_block )
                # the last volume
                self.fake_H[0, 0, -pt:, :, :] = \
                    (self.fake_H[0, 0, -pt:, :, :] * stitch_mask[0, 0, -pt:, :, :] +
                    fake_H_in_chunks[-1,...].float() * (2 - stitch_mask[0, 0, -pt:, :, :])) / 2
        self.netG.train()


    def optimize_parameters(self, step):
        self.optimizer_G.zero_grad()
        self.fake_H = self.netG(self.var_L)
        l_pix = self.l_pix_w * self.cri_pix(self.fake_H, self.real_H)
        with amp.scale_loss(l_pix, self.optimizer_G) as scale_loss:
            scale_loss.backward()
        self.optimizer_G.step()

        # set log
        self.log_dict['l_pix'] = l_pix.item()


    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self, data, maskOn=True, need_HR=True):
        out_dict = OrderedDict()
        out_dict['LR'] = self.var_L.detach()[0, 0].float().cpu()
        out_dict['SR'] = self.fake_H.detach()[0, 0].float().cpu()
        if maskOn:
            # the way we contructed mask it is 1 x 1 x depth x width x height
            mask = data['mask'].detach().float().cpu()[0, 0, :]
            out_dict['SR'] *= mask
        if need_HR:
            out_dict['HR'] = self.real_H.detach().float().cpu()[0, 0, :]
            if maskOn:
                out_dict['HR'] *= mask
        return out_dict

    def print_network(self):
        s, n = self.get_network_description(self.netG)
        if isinstance(self.netG, nn.DataParallel):
            net_struc_str = '{} - {}'.format(self.netG.__class__.__name__,
                                             self.netG.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.netG.__class__.__name__)

        logger.info('Network G structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
        logger.info(s)

    def load(self):
        load_path_G = self.opt['path']['pretrain_model_G']
        if load_path_G is not None:
            logger.info('Loading pretrained model for G [{:s}] ...'.format(load_path_G))
            self.load_network(load_path_G, self.netG)

    def save(self, iter_step):
        self.save_network(self.netG, 'G', iter_step)
