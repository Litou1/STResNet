import argparse
from models import dist_model as dm
from util import util
import numpy as np
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--path0', type=str, default='./imgs/ex_ref.png')
parser.add_argument('--path1', type=str, default='./imgs/ex_p0.png')
parser.add_argument('--use_gpu', action='store_true', help='turn on flag to use GPU')
opt = parser.parse_args()

## Initializing the model
model = dm.DistModel()
model.initialize(model='net-lin',net='vgg',use_gpu=opt.use_gpu)

# Load images
img0 = util.im2tensor(util.load_image(opt.path0), factor=1500./2.) # RGB image from [-1,1]
img1 = util.im2tensor(util.load_image(opt.path1), factor=1500./2.)

# Compute distance
dist01 = model.forward(img0,img1)
print('Distance: %.4f'%dist01)
