"""
transmorph new mind
"""

import os
import argparse
import numpy as np
import nibabel as nib
import torch
import time
from torchsummary import summary

# import voxelmorph with pytorch backend
os.environ['VXM_BACKEND'] = 'pytorch'
import voxelmorph as vxm
import cupy as cp
from scipy.ndimage import generate_binary_structure, binary_erosion
from scipy.ndimage.morphology import distance_transform_edt
from scipy.ndimage import _ni_support
import torch.nn.functional as F


# parse commandline args
parser = argparse.ArgumentParser()
parser.add_argument('--scansdir', help='pytorch model for nonlinear registration')
parser.add_argument('--labelsdir', help='test scan npz directory')
parser.add_argument('--model', help='pytorch model for nonlinear registration')
parser.add_argument('--labels', help='label lookup file in npz format')
parser.add_argument('--dataset', default='flare', help='dataset')

parser.add_argument('-g', '--gpu', help='GPU number(s) - if not supplied, CPU is used')
parser.add_argument('--multichannel', action='store_true', help='specify that data has multiple channels')
args = parser.parse_args()


# device handling
if args.gpu and (args.gpu != '-1'):
    device = 'cuda'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

else:
    device = 'cpu'
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

scansdir = args.scansdir
labelsdir = args.labelsdir
test_data = os.listdir(scansdir)
test_data = sorted(test_data, key=str.lower)

# load moving and fixed images
add_feat_axis = not args.multichannel

if args.dataset == 'mind':
    img_size = (160, 192, 160)
else:
    img_size = (128, 128, 96)

# model = vxm.iirp.MIRNet(img_size)
# best_model = torch.load(args.model)
# model.load_state_dict(best_model)
model = vxm.iirp.IIRPNet.load(args.model, device)

model.to(device)

summary(model)
model.eval()

# Use this to warp segments
trf = vxm.layers.SpatialTransformer(img_size, mode='nearest')
trf.to(device)
da = jnum = 0
dice_total = []
repeat_times = 0

for i in range(len(test_data) - 1):
    # print(i)
    atlas_dir = scansdir + '/' + test_data[i]
    labels_dir = labelsdir + '/' + test_data[i]
    atlas_vol = vxm.py.utils.load_volfile(atlas_dir, np_var='vol', add_batch_axis=True, add_feat_axis=add_feat_axis)
    atlas_vol1 = vxm.py.utils.load_volfile(atlas_dir, np_var='vol')
    atlas_seg = vxm.py.utils.load_volfile(labels_dir, np_var='seg')
    if i == 0:
        if args.dataset == 'mind':
            labels = np.load(args.labels)['labels']
            print('mind_label')
        else:
            labels = np.unique(atlas_seg)
            labels = labels[1:]
            print('flare_label')
        print(len(labels))
    for j in range(i+1, len(test_data)):
        repeat_times += 1
        moving_dir = scansdir + '/' + test_data[j]
        labels_dir = labelsdir + '/' + test_data[j]
        moving_vol = vxm.py.utils.load_volfile(moving_dir, np_var='vol', add_batch_axis=True, add_feat_axis=add_feat_axis)
        moving_seg = vxm.py.utils.load_volfile(labels_dir, np_var='seg', add_batch_axis=True, add_feat_axis=add_feat_axis)
        moving_seg1 = vxm.py.utils.load_volfile(labels_dir, np_var='seg')
        moving_vol1 = vxm.py.utils.load_volfile(moving_dir, np_var='vol')

        input_moving = torch.from_numpy(moving_vol).to(device).float().permute(0, 4, 1, 2, 3)
        input_fixed = torch.from_numpy(atlas_vol).to(device).float().permute(0, 4, 1, 2, 3)
        # predict and apply transform
        with torch.no_grad():
            warped_mov, warp1,ct = model(input_moving, input_fixed)
        print(ct)
        input_seg = torch.from_numpy(moving_seg).to(device).float().permute(0, 4, 1, 2, 3)
        warped_seg = trf(input_seg, warp1)
        warped_seg = warped_seg.detach().cpu().numpy().squeeze()
        overlap = vxm.py.utils.dice(warped_seg, atlas_seg, labels=labels)
        print(np.mean(overlap))

        dice_total.append(np.mean(overlap))

        df = warp1.detach().cpu().numpy().squeeze()
        df1 = df.transpose(1,2,3,0)
        jb = vxm.py.utils.jacobian_determinant(df1)
        jnum += np.sum(jb <= 0)


dice_total = np.array(dice_total)
jnum = jnum / repeat_times
print('Dice mean: %6.4f jnum: %6.4f ' % (dice_total.mean(), jnum))
