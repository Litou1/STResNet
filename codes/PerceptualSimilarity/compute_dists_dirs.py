import argparse
import os
from IPython import embed
from util import util
import models.dist_model as dm

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# parser.add_argument('--dir0', type=str, default='./imgs/ex_dir0')
# parser.add_argument('--dir1', type=str, default='./imgs/ex_dir1')
parser.add_argument('--dir0', type=str, default='/home/leihaowei/datasets/UCLA/png/val/k2_d100_st1.0')
parser.add_argument('--dir1', type=str, default='/home/leihaowei/MII/myrepo/SRGAN/results/SRGAN_denoise_003/val_num15_d10_d100_k2_st1')
parser.add_argument('--out', type=str, default='./imgs/SRGAN_denoise_003.txt')
parser.add_argument('--use_gpu', action='store_true', help='turn on flag to use GPU')
opt = parser.parse_args()

## Initializing the model
model = dm.DistModel()
model.initialize(model='net-lin',net='vgg',use_gpu=opt.use_gpu)

# crawl directories
f = open(opt.out,'w')

# files = os.listdir(opt.dir0)
files = []
for root, dirpath, fnames in sorted(os.walk(opt.dir0)):
    for fname in sorted(fnames):
        img_path = os.path.join(root, fname)
        # split path and combine
        img_path=os.path.join(os.path.split(os.path.split(img_path)[0])[-1], \
        	os.path.split(img_path)[-1])
        files.append(img_path)

avg_dist = 0.0
for file in files:
	if(os.path.exists(os.path.join(opt.dir1,file))):
		# Load images
		img0 = util.im2tensor(util.load_image(os.path.join(opt.dir0,file)), factor=1500./2.) # RGB image from [-1,1]
		img1 = util.im2tensor(util.load_image(os.path.join(opt.dir1,file)), factor=1500./2.)

		# Compute distance
		dist01 = model.forward(img0,img1)
		avg_dist += dist01
		print('%s: %.6f'%(file,dist01))
		f.writelines('%s: %.6f\n'%(file,dist01))
avg_dist = avg_dist / len(files)
f.writelines('average LPIPS: %.6f\n'%(avg_dist))
f.close()
