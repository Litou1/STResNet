import os
import glob
import torch
import torch.nn as nn
import math
import models.networks as networks
import copy

class BaseModel():
    def __init__(self, opt):
        self.opt = opt
        self.device = torch.device('cuda' if opt['gpu_ids'] is not None else 'cpu')
        self.is_train = opt['is_train']
        self.schedulers = []
        self.optimizers = []
        # define network and load pretrained models
        self.netG = networks.define_G(opt).to(self.device)
        self.print_network()
        self.load()

    def feed_train_data(self, data,  need_HR=True):
        self.var_L = data['LR'].to(self.device, non_blocking=True)  # LR
        if need_HR:
            self.real_H = data['HR'].to(self.device, non_blocking=True)  # HR

    def feed_test_data(self, data, need_HR=True):
        if need_HR:
            self.real_H = data['HR'].to(self.device, non_blocking=True)  # HR
        # basic block pixels
        opt_val = self.opt['datasets']['val']
        pt = opt_val['slice_size']
        # overlap pixels
        self.ot = opt_val['overlap_slice_size']
        self.nt = 1 + math.ceil( (data['LR'].size(2) - pt) / (pt - self.ot) )

        # reshape the whole volume into blocks of voxel of size pt x 512 x 512
        self.var_L = torch.empty(self.nt, 1, pt, data['LR'].size(3), data['LR'].size(4))\
            .to(self.device, non_blocking=True)
        if opt_val['full_volume']:
            self.var_L = data['LR']
        else:
            # n-1 volumes
            for i in range(0, self.nt - 1):
                self.var_L[i, :, :, :, :] = \
                    data['LR'][0, 0, i*(pt-self.ot):i*(pt-self.ot)+pt, :,:]
            # the last one
            self.var_L[-1, :, :, :, :] = \
                data['LR'][0, 0, -pt:, :, :]

    def half(self):
        if self.opt['precision'] == 'fp16':
            # copy fp32 model and convert to fp16
            self.netG_eval = copy.deepcopy(self.netG).half()
        else:
            self.netG_eval = self.netG
            
    def prepare_quant(self, loader):
        # PyTorch Static Quantization 
        # https://leimao.github.io/blog/PyTorch-Static-Quantization/
        fused_model = copy.deepcopy(self.netG)
        fused_model.eval()
        # fused conv3d + relu
        fused_model = torch.quantization.fuse_modules(fused_model.net, [["3", "4"],["5","6"]], inplace=True)
        for i in range(8):
            torch.quantization.fuse_modules(fused_model[1].res[i].res, [["0", "1"]], inplace=True)
        quantized_model = networks.define_Quant(model_fp32=fused_model)
        # config
        quantization_config = torch.quantization.get_default_qconfig("fbgemm")
        quantized_model.qconfig = quantization_config
        torch.quantization.prepare(quantized_model, inplace=True)
        # calibration
        data = next(iter(loader))  
        # get a small chunk for calibration
        _ = quantized_model(data['LR'][:,:,:32 ,:,:])
        quantized_model = torch.quantization.convert(quantized_model, inplace=True)
        self.netG_eval = quantized_model

    def optimize_parameters(self):
        pass

    def get_current_visuals(self):
        pass

    def get_current_losses(self):
        pass

    def print_network(self):
        pass

    def save(self, label):
        pass

    def load(self):
        pass

    def update_learning_rate(self):
        for scheduler in self.schedulers:
            scheduler.step()

    def get_current_learning_rate(self):
        return self.schedulers[0].get_lr()[0]

    def get_network_description(self, network):
        '''Get the string and total parameters of the network'''
        if isinstance(network, nn.DataParallel):
            network = network.module
        s = str(network)
        n = sum(map(lambda x: x.numel(), network.parameters()))
        return s, n

    def save_network(self, network, network_label, iter_step):
        save_filename = '{}_{}.pth'.format(iter_step, network_label)
        save_path = os.path.join(self.opt['path']['models'], save_filename)
        if isinstance(network, nn.DataParallel):
            network = network.module
        state_dict = network.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()
        # remove the previous files
        # old_files = glob.glob(os.path.join(self.opt['path']['models'], "*"+network_label+"*"))
        # for f in old_files:
        #     os.remove(os.path.join(self.opt['path']['models'], f))
        # save the new file
        torch.save(state_dict, save_path)

    def load_network(self, load_path, network, strict=True):
        if isinstance(network, nn.DataParallel):
            network = network.module
        network.load_state_dict(torch.load(load_path), strict=strict)

    def save_training_state(self, epoch, iter_step):
        '''Saves training state during training, which will be used for resuming'''
        state = {'epoch': epoch, 'iter': iter_step, 'schedulers': [], 'optimizers': []}
        for s in self.schedulers:
            state['schedulers'].append(s.state_dict())
        for o in self.optimizers:
            state['optimizers'].append(o.state_dict())
        save_filename = '{}.state'.format(iter_step)
        save_path = os.path.join(self.opt['path']['training_state'], save_filename)
        # remove the previous file
        old_files = os.listdir(self.opt['path']['training_state'])
        for f in old_files:
            os.remove(os.path.join(self.opt['path']['training_state'],f))
        # save the new file
        torch.save(state, save_path)

    def resume_training(self, resume_state):
        '''Resume the optimizers and schedulers for training'''
        resume_optimizers = resume_state['optimizers']
        resume_schedulers = resume_state['schedulers']
        assert len(resume_optimizers) == len(self.optimizers), 'Wrong lengths of optimizers'
        assert len(resume_schedulers) == len(self.schedulers), 'Wrong lengths of schedulers'
        for i, o in enumerate(resume_optimizers):
            self.optimizers[i].load_state_dict(o)
        for i, s in enumerate(resume_schedulers):
            self.schedulers[i].load_state_dict(s)
