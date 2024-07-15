# IIRP-Net
This is the official pytorch implementation of "IIRP-Net: Iterative Inference Residual Pyramid Network for Enhanced Image
Registration" (CVPR 2024), written by Tai Ma, Suwei Zhang, Jiafeng Li and Ying Wen. Paper link: https://openaccess.thecvf.com/content/CVPR2024/html/Ma_IIRP-Net_Iterative_Inference_Residual_Pyramid_Network_for_Enhanced_Image_Registration_CVPR_2024_paper.html
![image](https://github.com/Torbjorn1997/IIRP-Net/blob/main/001.png)
## Environment
We reimplemented the code on pytorch 1.13 and python 3.7.15. 
## Dataset
We performed retraining，validation and testing on the Mindboggle dataset. 

We provide pre-trained models on the Mindboggle dataset, trained with two subsets, NKI-RS and NKI-TRT, with images cropped to the size of (160, 192, 160).

## Citation
If you use the code in your research, please cite:
```bibtex
@InProceedings{Ma_2024_CVPR,
    author    = {Ma, Tai and Zhang, Suwei and Li, Jiafeng and Wen, Ying},
    title     = {IIRP-Net: Iterative Inference Residual Pyramid Network for Enhanced Image Registration},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2024},
    pages     = {11546-11555}
}
```
The overall framework of the code and the Swin Transformer module are based on [VoxelMorph](https://github.com/voxelmorph/voxelmorph) , whose contributions are greatly appreciated.
## Test
We provide the pre-trained model and two images for testing from the MMRR subset of the Mindboggle dataset. You can test it with the following code:
For RP-Net
```code
python test.py --scansdir  data/vol --labelsdir  data/seg --dataset mind --labels  data/label_mind.npz --model model/mind.pt --gpu 0
```
The test results are:
![image](https://github.com/user-attachments/assets/00545f68-0fce-4fbb-9a1a-9cf9597dd5c5)
For IIRP-Net
```code
python test_iirp.py --scansdir  data/vol --labelsdir  data/seg --dataset mind --labels  data/label_mind.npz --model model/mind.pt --gpu 0
```
The test results are:
![image](https://github.com/user-attachments/assets/3889761c-07e6-41a6-9b1a-20aa902e9b16)
