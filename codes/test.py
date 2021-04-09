import os, glob, time
import logging
import argparse
from collections import OrderedDict
import numpy as np
import torch
import options.options as option
import utils.util as util
from data import create_dataset, create_dataloader
from models import create_model
import random


def main():
    # options
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, required=True, help='Path to options JSON file.')
    opt = option.parse(parser.parse_args().opt, is_train=False)
    util.mkdirs((path for key, path in opt['path'].items() if not key == 'pretrain_model_G'))
    opt = option.dict_to_nonedict(opt)

    util.setup_logger(None, opt['path']['log'], 'test.log', level=logging.INFO, screen=True)
    logger = logging.getLogger('base')
    logger.info(option.dict2str(opt))

    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = False

    test_loaders = []
    for phase, dataset_opt in opt['datasets'].items():
        # dataset_opt['uids'] = uids[85:] if dataset_opt['data_type'] == 'h5' else uids
        test_set = create_dataset(dataset_opt)
        test_loader = create_dataloader(test_set, dataset_opt)
        logger.info('Number of test volumes in [{:s}]: {:d}'.format(dataset_opt['name'], len(test_set)))
        test_loaders.append(test_loader)

    # Create model
    model = create_model(opt)
    # copy net and convert to fp16 if necessary
    model.half()
    # create pdist model vgg
    pdist_model = util.create_pdist_model(use_gpu=opt['gpu_ids'] is not None)
    if opt["precision"] == 'int8':
        model.prepare_quant(test_loaders[0])
    for test_loader in test_loaders:
        test_set_name = test_loader.dataset.opt['name']
        logger.info('\nTesting [{:s}]...'.format(test_set_name))
        results_dir = os.path.join(opt['path']['results_root'], test_set_name)
        util.mkdir(results_dir)

        pnsr_results = OrderedDict()
        ssim_results = OrderedDict()
        pdist_results = OrderedDict()
        total_t = 0.
        for i, data in enumerate(test_loader):
            need_HR = False if test_loader.dataset.opt['dataroot_HR'] is None else True
            has_mask = False if test_loader.dataset.opt['maskroot_HR'] is None else True

            logger.info('start inference...')
            model.feed_test_data(data, need_HR=need_HR)
            t0 = time.time()
            model.test(data)  # test
            if opt['gpu_ids']:
                torch.cuda.synchronize()
            t1 = time.time()
            t = t1 - t0
            total_t += t
            logger.info('inference time: {:.3f}'.format(t))
            # get cpu numpy from cuda tensor
            visuals = model.get_current_visuals(data, maskOn=has_mask, need_HR=need_HR)
            # save volume data
            patient_id = data['uid'][0]
            vol_path = os.path.join(results_dir, patient_id)
            LR_spacings = [x.item() for x in data['spacings']]
            if 'nrrd' in opt['result_format']:
                logger.info('saving nnrd...')
                sr_vol = util.tensor2img(visuals['SR'], out_type=np.uint16) 
                util.save_vol(opt, LR_spacings, vol_path + '.nrrd', sr_vol)
            if 'dicom' in opt['result_format']:
                logger.info('saving dicoms...') 
                sr_vol = util.tensor2img(visuals['SR'], out_type=np.int16, intercept = -1000) 
                util.save_dicoms(opt, LR_spacings, vol_path, sr_vol)
            if  opt['result_format']!='nrrd' and opt['result_format']!='dicom':
                raise NotImplementedError('supported output format: nrrd or dicom')

            #initialize dictionary with empty {}
            pnsr_results[patient_id] = {}
            ssim_results[patient_id] = {}
            pdist_results[patient_id] = {}
            # need_HR = False
            if need_HR:
                def _calculate_metrics(sr_vol, gt_vol, view='xy'):
                    sum_psnr = 0.
                    sum_ssim = 0.
                    sum_pdist = 0.
                    # [D,H,W]
                    num_val = 0 # psnr could be inf at xz or yz (near edges), will not calculate
                    for i, vol in enumerate(zip(sr_vol, gt_vol)):
                        sr_img, gt_img = vol[0], vol[1]
                        # calculate PSNR and SSIM
                        # range is assume to be [0,255] so  have to scale back from 1500 to 255 float64
                        crop_size = round(opt['scale'])
                        cropped_sr_img = sr_img[crop_size:-crop_size, crop_size:-crop_size]\
                                             .astype(np.float64) / 1500. * 255.
                        cropped_gt_img = gt_img[crop_size:-crop_size, crop_size:-crop_size]\
                                             .astype(np.float64) / 1500. * 255.

                        psnr = util.calculate_psnr(cropped_sr_img, cropped_gt_img)
                        ssim = util.calculate_ssim(cropped_sr_img, cropped_gt_img)
                        if opt['datasets']['val']['get_pdist']:
                            pdist = util.calculate_pdist(pdist_model, cropped_sr_img, cropped_gt_img)
                        else:
                            pdist = float('nan')
                        
                        if psnr != float('inf'):
                            num_val += 1
                            sum_psnr += psnr
                            sum_ssim += ssim
                            sum_pdist += pdist
                        logger.info('{:20s} - {:3d}- PSNR: {:.6f} dB; SSIM: {:.6f}; pdist: {:.6f}.'\
                                    .format(patient_id, i+1, psnr, ssim, pdist))

                    pnsr_results[patient_id][view] = sum_psnr / num_val
                    ssim_results[patient_id][view] = sum_ssim / num_val
                    pdist_results[patient_id][view] = sum_pdist / num_val
                    return pnsr_results, ssim_results, pdist_results
                
                sr_vol = util.tensor2img(visuals['SR'], out_type=np.uint16) 
                gt_vol = util.tensor2img(visuals['HR'], out_type=np.uint16)  # uint16 range [0,1500]
                min_depth = min(sr_vol.shape[0], gt_vol.shape[0])
                sr_vol = sr_vol[:min_depth,...]
                gt_vol = gt_vol[:min_depth,...] # make sure they have the same depth
                # [H W] axial view
                _calculate_metrics(sr_vol, gt_vol, view='xy')
                # [D W] coronal view
                _calculate_metrics(sr_vol.transpose(1, 0, 2), gt_vol.transpose(1, 0, 2), view='xz')
                # [D H] sagittal view
                _calculate_metrics(sr_vol.transpose(2, 0, 1), gt_vol.transpose(2, 0, 1), view='yz')

            else:
                logger.info(patient_id)


        if need_HR:  # metrics
            # print result dictionary
            util.print_metrics(logger, 'test PSNR', pnsr_results)
            util.print_metrics(logger, 'test SSIM', ssim_results)
            util.print_metrics(logger, 'test pdist', pdist_results)
        logger.info('average inference time: {:.3f}'.format(total_t/len(test_loader)))

if __name__ == '__main__':
    main()