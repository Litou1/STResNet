import os, time
import math
from datetime import datetime
import numpy as np
import cv2
import random
import torch
import logging
import nrrd
import json
# from pydicom.dataset import Dataset, FileDataset
import SimpleITK as sitk
from PerceptualSimilarity.models import dist_model as dm


####################
# miscellaneous
####################
def get_timestamp():
    return datetime.now().strftime('%y%m%d-%H%M%S')


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def mkdirs(paths):
    if isinstance(paths, str):
        mkdir(paths)
    else:
        for path in paths:
            mkdir(path)


def mkdir_and_rename(path):
    if os.path.exists(path):
        new_name = path + '_archived_' + get_timestamp()
        print('Path already exists. Rename it to [{:s}]'.format(new_name))
        logger = logging.getLogger('base')
        logger.info('Path already exists. Rename it to [{:s}]'.format(new_name))
        os.rename(path, new_name)
    os.makedirs(path)


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_logger(logger_name, root, phase, level=logging.INFO, screen=False):
    '''set up logger'''
    l = logging.getLogger(logger_name)
    formatter = logging.Formatter(
        '%(asctime)s.%(msecs)03d - %(levelname)s: %(message)s', datefmt='%y-%m-%d %H:%M:%S')
    log_file = os.path.join(root, phase + '_{}.log'.format(get_timestamp()))
    fh = logging.FileHandler(log_file, mode='w')
    fh.setFormatter(formatter)
    l.setLevel(level)
    l.addHandler(fh)
    if screen:
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        l.addHandler(sh)

def print_metrics(logger, metric_name, results):
    means = {}
    for view in list(results[list(results.keys())[0]].keys()):
        count = 0
        sum = 0.
        for k, v in results.items():
            count += 1
            sum += v[view]
            logger.info('{} for {} in {} view : {:.6f}'.format(metric_name, k, view, v[view]))
        means[view] = sum/count
        logger.info('Average {} in {} view : {:.6f}'.format(metric_name, view, means[view]))
    return means


####################
# image operations
####################
def tensor2img(tensor, out_type=np.uint8, min_max=(0, 1), intercept=-1024):
    '''
    Converts a torch Tensor into an image Numpy array
    Input: 3D(D,H,W)
    Output: 3D(H,W,D), np.uint8 (default), [0,1500] uint16, [-1000, 500] int16
    intercept is used when out_type is set to int16
    '''
    tensor = tensor.cpu().clamp_(*min_max)  # clamp
    tensor = (tensor - min_max[0]) / (min_max[1] - min_max[0])  # to range [0,1]
    img_np = tensor.numpy()
    if out_type == np.uint8:
        img_np = (img_np * 255.0).round()
        # Important. Unlike matlab, numpy.uint8() WILL NOT round by default.
    if out_type == np.uint16:
        img_np = (img_np * 1500.0).round()
    if out_type == np.int16:
        img_np = (img_np * 1500.0).round() + intercept 
    return img_np.astype(out_type)


def img2tensor(image, cent=1., factor=255./2.):
    '''
    :Input 3D(H,W,D):
    :Output (1,D,H,W) [-1,1]
    '''
    return torch.Tensor((image / factor - cent)[:, :, :, np.newaxis].transpose((3, 2, 0, 1)))


class ImgToTensor(object):
    '''
    call this function to covnvert h5 uint16 (D,H,W)
    to torch tensor float in range [0,1]
    for dataloader use, inplace operation
    '''
    def __call__(self, sample, raw_data_range = 1500.):
        img = torch.from_numpy(np.array(sample, np.float32, copy=False))
        return img.float().div_(raw_data_range)  # range is from 0 to 1500 for png
        # return img.float().div_(300.)  # range is from 0 to 1500 for png

def read_config(config_path):
    config_path = os.path.join(config_path) 
    with open(config_path, 'r') as f:
        config = json.load(f)
    meta_data = {}
    meta_data['Spacing'] = [float(i) for i in config['Spacing'].split()]
    meta_data['Orientation'] = [float(i) for i in config['Orientation'].split()] 
    meta_data['Origin'] = [float(i) for i in config['Origin'].split()]
    return meta_data

def save_vol(opt, spacings, volume_path, volume):
    # nrrd accepts in [W,H,D] format
    # [D,H,W]==>[W,H,D]
    # no need to transpose, just add index_order='C'
    # volume = np.transpose(volume, (2, 1, 0))

    copy_spacings = spacings.copy()
    copy_spacings[2] /= opt['scale'] 
    # uid = os.path.splitext(os.path.basename(volume_path))[0]
    # if opt['datasets']['val']['data_type'] == 'h5':
    #     # get spacing from json file
    #     config_path = os.path.join(opt['datasets']['val']['dataroot_LR'], uid + '.json')
    #     meta_data = read_config(config_path)
    #     spacings = meta_data['Spacing']
    #     orientations = meta_data['Orientation']
    #     # nrrd requires nested list for orientations
    #     # https://readthedocs.org/projects/pynrrd/downloads/pdf/latest/
    #     # http://teem.sourceforge.net/nrrd/format.html#spacedirections
    #     orientations = [orientations[i:i+3] for i in range(0, len(orientations), 3)]
    #     origin = meta_data['Origin']         
    #     spacings[2] /= opt['scale'] 
    # elif opt['datasets']['val']['data_type'] == 'dicom':
    #     # TODO: implement save according to input dicom
    #     spacings[2] /= opt['scale'] 
    # else: 
    #     raise NotImplementedError('supported output format: nrrd or dicom')

    if opt['datasets']['val']['data_type'] == 'dicom':
        header = {'units': ['mm', 'mm', 'mm'], 'spacings': copy_spacings}
        nrrd.write(volume_path, volume, header, index_order='C')
        return
    header = {'units': ['mm', 'mm', 'mm'], 'spacings': copy_spacings} 
                                        #    'space directions': orientations,
                                        #    'space origin': origin}
    nrrd.write(volume_path, volume, header, index_order='C')
    # nrrd.write(volume_path, volume, index_order='C')


def save_dicoms(opt, spacings, volume_path, volume):
    mkdir(volume_path)
    copy_spacings = spacings.copy()
    copy_spacings[2] /= opt['scale'] 
    # if opt['datasets']['val']['data_type'] == 'h5':     
    #     config_path =  os.path.join(opt['datasets']['val']['dataroot_LR'], uid + '.json')
    #     meta_data = read_config(config_path)
    #     spacings = meta_data['Spacing']
    #     spacings[2] /= opt['scale'] 
    #     orientations = meta_data['Orientation']
    #     orientations = [orientations[i:i+3] for i in range(0, len(orientations), 3)]
    #     origin = meta_data['Origin']  
    # elif opt['datasets']['val']['data_type'] == 'dicom':
    #     spacings[2] /= opt['scale'] 
    # else: 
    #     raise NotImplementedError('supported output format: nrrd or dicom')
    write_dicom(copy_spacings, volume, volume_path)            
    print("written as dcm completed")
    # for i, pixel_array in enumerate(vol): 
    #     write_dicom(i+1, opt, meta_data, pixel_array, volume_path)


def write_dicom(spacings, new_arr, path):
    new_img = sitk.GetImageFromArray(new_arr)
    new_img.SetSpacing(spacings)
    
    writer = sitk.ImageFileWriter()
    # Use the study/series/frame of reference information given in the meta-data
    # dictionary and not the automatically generated information from the file IO
    writer.KeepOriginalImageUIDOn()

    modification_time = time.strftime("%H%M%S")
    modification_date = time.strftime("%Y%m%d")

    # Copy some of the tags and add the relevant tags indicating the change.
    # For the series instance UID (0020|000e), each of the components is a number, cannot start
    # with zero, and separated by a '.' We create a unique series ID using the date and time.
    # tags of interest:
    direction = new_img.GetDirection()
    series_tag_values = [("0008|0031",modification_time), # Series Time
                    ("0008|0021",modification_date), # Series Date
                    ("0008|0008","DERIVED\\SECONDARY"), # Image Type
                    ("0020|000e", "1.2.826.0.1.3680043.2.1125."+modification_date+".1"+modification_time), # Series Instance UID
                    ("0020|0037", '\\'.join(map(str, (direction[0], direction[3], direction[6],# Image Orientation (Patient)
                                                    direction[1],direction[4],direction[7])))),
                    ("0008|103e", "Normalized image"), # series description
                    ("0020|000d", "1.2.826.0.1.3680043.2.1125."+modification_date+".1"+modification_time), # study instance UID
                    ("0008|0020", modification_date), # study date
                    ("0008|0030", modification_time),  # study time
                    # Setting the type to CT preserves the slice location.
                    # set the type to CT so the thickness is carried over
                    ("0008|0060", "CT"),
                    ("0028|1050", "-600"), # WindowCenter
                    ("0028|1051", "1500"), # WindowWidth
                    ("0028|1054", "HU"), # RescaleType
                    ("0018|0050", str(spacings[2]))  # slice thickness
                    ]  

    # Write slices to output directory
    list(map(lambda i: writeSlices(writer, i, series_tag_values, new_img, path), range(new_img.GetDepth())))

def writeSlices(writer, index, series_tag_values, new_img, path):
    # DicomSeriesFromArray
    #https://simpleitk.readthedocs.io/en/next/Examples/DicomSeriesFromArray/Documentation.html
    image_slice = new_img[:,:,index]

    # Tags shared by the series.
    list(map(lambda tag_value: image_slice.SetMetaData(tag_value[0], tag_value[1]), series_tag_values))

    # Slice specific tags.
    image_slice.SetMetaData("0008|0012", time.strftime("%Y%m%d")) # Instance Creation Date
    image_slice.SetMetaData("0008|0013", time.strftime("%H%M%S")) # Instance Creation Time

    # (0020, 0032) image position patient determines the 3D spacing between slices.
    pos = [str(i) for i in new_img.TransformIndexToPhysicalPoint((0,0,index))]
    image_slice.SetMetaData("0020|0032", '\\'.join(pos)) # Image Position (Patient)
    image_slice.SetMetaData("0020|1041", pos[2]) # Slice location
    image_slice.SetMetaData("0020|0013", str(index+1)) # Instance Number

    # No need to set intercept/slope in ITK
    # RescaleIntercept 
    # image_slice.SetMetaData("0028|1052", "-1000")
    # RescaleSlope
    # image_slice.SetMetaData("0028|1053", "1")
    
    # Write to the output directory and add the extension dcm, to force writing in DICOM format.
    writer.SetFileName(os.path.join(path, '{0:0=3d}.dcm'.format(index+1)))
    writer.Execute(image_slice)

# need some work to get it working using pydicom
def write_pydicom(index, opt, meta_data, pixel_array, path):
    fullpath = os.path.join(path, '{0:0=3d}.dcm'.format(index))
    pixel_array = pixel_array.astype(np.int16)

    file_meta = Dataset()
    file_meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.2'
    file_meta.MediaStorageSOPInstanceUID = "1.2.840.113654.2.55.102869246940549636091903990697209280630"
    file_meta.ImplementationClassUID = '1.2.40.0.13.1.1.1'
    file_meta.TransferSyntaxUID = '1.2.840.10008.1.2.1' 

    ds = FileDataset(fullpath, {},file_meta = file_meta, preamble=b"\0"*128)

    # ds.SOPClassUID = '1.2.840.10008.5.1.4.1.1.2'
    # ds.SOPInstanceUID = "1.2.840.113654.2.55.102869246940549636091903990697209280630"
    ds.SeriesDescription = 'Normalized images'
    ds.SeriesInstanceUID = '1.2.840.113654.2.55.102869246940549636091903990697209280630'
    # ds.StudyInstanceUID =  '1.2.840.113654.2.55.102869246940549636091903990697209280630'
    # ds.StudyID = ''
    # ds.ContentDate = str(datetime.date.today()).replace('-','')
    # ds.ContentTime = str(time.time()) #milliseconds since the epoch
    ds.Modality = "CT"
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.PixelRepresentation = 0
    ds.HighBit = 11
    ds.BitsStored = 12
    ds.BitsAllocated = 16
    ds.Columns, ds.Rows = pixel_array.shape
    ds.SeriesNumber = 1
    ds.AcqusitionNumber = 1
    
    ds.RescaleSlope = "1"
    ds.RescaleIntercept = "-1000"
    ds.RescaleType  = "HU"
    ds.WindowCenter = "-600"
    ds.WindowWidth = "1500"    
    ds.InstanceNumber = index


    spacings = meta_data['Spacing']
    spacings[2] /= opt['scale'] 
    # orientations = meta_data['Orientation']
    # origin = meta_data['Origin']  

    # ds.ImagePositionPatient = origin       
    # ds.SliceLocation = origin[2] - (index - 1) * spacings[2]   
    # ds.ImageOrientationPatient = orientations[:6]
    ds.SliceThickness = spacings[2] + 1e-10 
    ds.PixelSpacing = [spacings[0], spacings[1]]
    ds.PixelData = pixel_array.tobytes()
    ds.save_as(fullpath)
    

####################
# metric
####################
def calculate_psnr(img1, img2):
    # img1 and img2 have range [0, 1]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))


def ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2
    # C1 = (0.01)**2
    # C2 = (0.03)**2
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calculate_ssim(img1, img2):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')


def create_pdist_model(network='vgg', use_gpu=True):
    '''
    percetual similarity metric
    https://github.com/richzhang/PerceptualSimilarity
    :param network can be vgg or alex
    '''
    ## Initializing the model
    model = dm.DistModel()
    model.initialize(model='net-lin', net=network, use_gpu=use_gpu)
    return model


def calculate_pdist(model, img1, img2):
    # expand channel to 3 dim for vgg
    img1 = np.expand_dims(img1, axis=2)
    img1 = np.repeat(img1, 3, axis=2)
    img2 = np.expand_dims(img2, axis=2)
    img2 = np.repeat(img2, 3, axis=2)
    #  image from [-1,1]
    img1 = img2tensor(img1)
    img2 = img2tensor(img2)

    # Compute distance
    with torch.no_grad():
        dist = model.forward(img1, img2)
    return dist[0]
