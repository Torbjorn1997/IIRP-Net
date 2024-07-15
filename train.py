import os
import glob
import sys
import random
import time
import torch
import numpy as np
# import scipy.ndimage
from argparse import ArgumentParser
from torchsummary import summary

os.environ['VXM_BACKEND'] = 'pytorch'
import voxelmorph as vxm

def train(datadir,
          model_dir,
          load_model,
          gpu,
          initial_epoch,
          epochs,
          steps_per_epoch,
          batch_size,
          atlas=False,
          bidir=False):

    train_vol_names = glob.glob(os.path.join(datadir, '*.nii.gz'))
    random.shuffle(train_vol_names)  # shuffle volume list
    assert len(train_vol_names) > 0, 'Could not find any training data'

    # no need to append an extra feature axis if data is multichannel
    add_feat_axis = True

    generator = vxm.generators.scan_to_scan(train_vol_names, batch_size=batch_size, bidir=bidir,
                                                add_feat_axis=add_feat_axis)

    # extract shape from sampled input
    inshape = next(generator)[0][0].shape[1:-1]

    os.makedirs(model_dir, exist_ok=True)


    # prepare odel folder
    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)

    device = 'cuda'
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu

    torch.backends.cudnn.deterministic = True

    # prepare the model
    model = vxm.iirp.RPNet(inshape)
    model.to(device)
    summary(model)
    if load_model != False:
        print('loading', load_model)
        best_model = torch.load(load_model)
        model.load_state_dict(best_model)

    # set optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # prepare losses
    Losses = [vxm.losses.NCC().loss, vxm.losses.Grad_2('l2').loss]
    Weights = [1.0, 1.0]

    # training/validate loops
    for epoch in range(initial_epoch, epochs):
        start_time = time.time()
        
        # training
        model.train()
        train_losses = []
        train_total_loss = []
        for step in range(steps_per_epoch):
            
            # generate inputs (and true outputs) and convert them to tensors
            inputs, labels = next(generator)
            # inputs = [torch.from_numpy(d).to(device).float() for d in inputs]
            # labels = [torch.from_numpy(d).to(device).float() for d in labels]
            inputs = [torch.from_numpy(d).to(device).float().permute(0, 4, 1, 2, 3) for d in inputs]  # 其实包括了俩
            labels = [torch.from_numpy(d).to(device).float().permute(0, 4, 1, 2, 3) for d in labels]  # 一个
            source = inputs[0]
            target = inputs[1]
            # run inputs through the model to produce a warped image and flow field
            pred = model(source, target)

            # calculate total loss
            loss = 0
            loss_list = []
            for i, Loss in enumerate(Losses):
                curr_loss = Loss(pred[i], target) * Weights[i]
                loss_list.append(curr_loss.item())
                loss += curr_loss
            train_losses.append(loss_list)
            train_total_loss.append(loss.item())

            # backpropagate and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # print epoch info
        epoch_info = 'Epoch %d/%d' % (epoch + 1, epochs)
        time_info = 'Total %.2f sec' % (time.time() - start_time)
        train_losses = ', '.join(['%.4f' % f for f in np.mean(train_losses, axis=0)])
        train_loss_info = 'Train loss: %.4f  (%s)' % (np.mean(train_total_loss), train_losses)
        print(' - '.join((epoch_info, time_info, train_loss_info)), flush=True)
    
        # save model checkpoint
        if (epoch+1) % 10 == 0:
            torch.save(model.state_dict(), os.path.join(model_dir, '%04d.pt' % (epoch+1)))
    

if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument('datadir', help='base data directory')
    parser.add_argument('--model-dir', default='models', help='model output directory (default: models)')
    parser.add_argument('--load-model', default=False, help='optional model file to initialize with')
    parser.add_argument('--gpu', default='0', help='GPU ID number(s), comma-separated (default: 0)')
    parser.add_argument('--initial-epoch', type=int, default=0, help='initial epoch number (default: 0)')
    parser.add_argument('--epochs', type=int, default=1000, help='number of training epochs (default: 1500)')
    parser.add_argument('--steps-per-epoch', type=int, default=100, help='frequency of model saves (default: 100)')
    parser.add_argument('--batch-size', type=int, default=1, help='batch size (default: 1)')

    args = parser.parse_args()
    train(**vars(args))

