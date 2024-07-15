import os
import sys
import glob
import numpy as np
import scipy


from . import py


def volgen(
        vol_names,
        batch_size=1,
        return_segs=False,
        return_warp=False,
        np_var='vol',
        pad_shape=None,
        resize_factor=1,
        add_feat_axis=True
):
    """
    Base generator for random volume loading. Volumes can be passed as a path to
    the parent directory, a glob pattern or a list of file paths. Corresponding
    segmentations are additionally loaded if return_segs is set to True. If
    loading segmentations, npz files with variable names 'vol' and 'seg' are
    expected.

    Parameters:
        vol_names: Path, glob pattern or list of volume files to load.
        batch_size: Batch size. Default is 1.
        return_segs: Loads corresponding segmentations. Default is False.
        np_var: Name of the volume variable if loading npz files. Default is 'vol'.
        pad_shape: Zero-pads loaded volumes to a given shape. Default is None.
        resize_factor: Volume resize factor. Default is 1.
        add_feat_axis: Load volume arrays with added feature axis. Default is True.
    """

    # convert glob path to filenames
    if isinstance(vol_names, str):
        if os.path.isdir(vol_names):
            vol_names = os.path.join(vol_names, '*')
        vol_names = glob.glob(vol_names)

    while True:
        # generate [batchsize] random image indices
        indices = np.random.randint(len(vol_names), size=batch_size)
        # load volumes and concatenate
        load_params = dict(np_var=np_var, add_batch_axis=True, add_feat_axis=add_feat_axis, pad_shape=pad_shape,
                           resize_factor=resize_factor)
        load_params_1 = dict(np_var=np_var, add_batch_axis=True, add_feat_axis=0, pad_shape=pad_shape,
                           resize_factor=resize_factor)
        imgs = [py.utils.load_volfile(vol_names[i], **load_params) for i in indices]
        vols = [np.concatenate(imgs, axis=0)]

        # optionally load segmentations and concatenate
        if return_segs:
            load_params['np_var'] = 'seg'  # be sure to load seg
            segs = [py.utils.load_volfile(vol_names[i], **load_params) for i in indices]
            vols.append(np.concatenate(segs, axis=0))
        if return_warp:
            load_params_1['np_var'] = 'warp'  # be sure to load seg
            warp = [py.utils.load_volfile(vol_names[i], **load_params_1) for i in indices]
            vols.append(np.concatenate(warp, axis=0))

        yield tuple(vols)

def scan_to_scan(vol_names, bidir=False, batch_size=1, prob_same=0, no_warp=False, **kwargs):
    """
    Generator for scan-to-scan registration.

    Parameters:
        vol_names: List of volume files to load.
        bidir: Yield input image as output for bidirectional models. Default is False.
        batch_size: Batch size. Default is 1.
        prob_same: Induced probability that source and target inputs are the same. Default is 0.
        no_warp: Excludes null warp in output list if set to True (for affine training). Default if False.
        kwargs: Forwarded to the internal volgen generator.
    """
    zeros = None
    gen = volgen(vol_names, batch_size=batch_size, **kwargs)
    while True:
        scan1 = next(gen)[0]
        scan2 = next(gen)[0]

        # some induced chance of making source and target equal
        if prob_same > 0 and np.random.rand() < prob_same:
            if np.random.rand() > 0.5:
                scan1 = scan2
            else:
                scan2 = scan1

        # cache zeros
        if not no_warp and zeros is None:
            shape = scan1.shape[1:-1]
            zeros = np.zeros((batch_size, *shape, len(shape)))

        invols = [scan1, scan2]
        outvols = [scan2, scan1] if bidir else [scan2]
        if not no_warp:
            outvols.append(zeros)

        yield (invols, outvols)


