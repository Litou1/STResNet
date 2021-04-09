'''create dataset and dataloader'''
import logging
import torch.utils.data

def create_dataloader(dataset, dataset_opt):
    '''create dataloader '''
    phase = dataset_opt['phase']
    if phase == 'train':
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=dataset_opt['batch_size'],
            shuffle=dataset_opt['use_shuffle'],
            num_workers=dataset_opt['n_workers'],
            drop_last=True,
            pin_memory=True)
    else:
        return torch.utils.data.DataLoader(
            dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)


def create_dataset(dataset_opt):
    '''create dataset'''
    if dataset_opt['data_type'] == 'h5':
        from data.h5Dataset import h5Dataset as D
    elif dataset_opt['data_type'] == 'dicom':
        from data.dcmDataset import dcmDataset as D
    # load uids in this phase    
    with open(dataset_opt['uids_path'], 'r') as f:
        lines = f.readlines()
        dataset_opt['uids'] = [l.rstrip() for l in lines]
    # print(dataset_opt['uids'])    
    dataset = D(dataset_opt)
    logger = logging.getLogger('base')
    logger.info('Dataset [{:s} - {:s}] is created.'.format(dataset.__class__.__name__,
                                                           dataset_opt['name']))
    return dataset
