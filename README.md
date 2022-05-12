# Pytorch-GAN-implementation
Collection of PyTorch implementations of GAN Integration.

I've tried to replicate the original paper as closely as possible. 



## Table of Contents
  
  * [Overview](#overview) 
  * [Implementations](#implementations)
    + [CycleGAN](#cyclegan)
      


## Overview

Folder explanation

```
.
├── configs      # Training options
├── data         # dataset & dataloader 
├── datasets     # datasets
├── model        # orginizing generator and discriminator even training process
└── utils        # utiliy python file 

```

## Prerequisites

 * System
      + Windows / Linux
      + CPU or GPU(Single/Multi)
      + Python 3
      
 * Libraries
      + Python>=3.6
      + PyTorch>=1.4 
      + torchvision>=0.5.0
      + numpy
      + yaml

## Implementations

### CycleGAN

<b>Title of Paper</b> : Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks

<b>Authors</b> : Jun-Yan Zhu(first author), Taesung Park(first author), Phillip Isola, Alexei A. Efros

[[Offical Paper]](https://openaccess.thecvf.com/content_ICCV_2017/papers/Zhu_Unpaired_Image-To-Image_Translation_ICCV_2017_paper.pdf) [[Arxiv]](https://arxiv.org/abs/1703.10593)

[[Official Pytorch Code]](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)

#### Datasets

```
.
├── dataset                   
|   ├── <dataset_name>         # i.e. horse2zebra
|   |   ├── train              # Training
|   |   |   ├── A              # Contains domain A images (i.e. horse)
|   |   |   └── B              # Contains domain B images (i.e. zebra)
|   |   └── test               # Testing
|   |   |   ├── A              # Contains domain A images (i.e. horse)
|   |   |   └── B              # Contains domain B images (i.e. zebra)
```

First, download and setup a dataset.

<b>Download methods</b>

1) Run below code file which is the same bash code of the [Official CycleGAN Pytorch Code](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/datasets/download_cyclegan_dataset.sh). (e.g. horse2zebra)

```
bash ./dataset/download_cyclegan_dataset.sh horse2zebra
```

2) Download from [UC Berkeley's repository](http://efrosgans.eecs.berkeley.edu/cyclegan/datasets)

Dataset will be placed in dataset folder

Second, checking whether the dataset directory path is different in configs/cyclegan.yaml. If there is a change, then type the path not only training but also testing.

#### Train

To train from the scratch, check the configs/cyclegan.yaml file for confirming the location of Path.
After checking, type the below command.

```
$ python3 train.py --opt configs/cyclegan.yaml
```

#### Test

To test the code with the pretrained models, 

1) Check the configs/cyclegan.yaml file for confirming the location of two test datasets and 2 pretrained model pt/pth files. (In CycleGAN, there are two datasets.)

2) Place the pth/pt file in pretrain_model_dir path of configs/cyclegan.yaml file. (Downloading the pretrained files are in below session.)

3) Type the below command.

```
$ python3 test.py --opt configs/cyclegan.yaml
```

#### Pretrained model

[[CycleGAN Pretrained pth]](https://drive.google.com/file/d/1FC0BeDzc8plkkvB69hcY5N28U1mDPssr/view?usp=sharing)

pth zip file include "160_net_G_AtoB.pth" and "160_net_G_BtoA.pth"

#### Results

The model was trained on A:Horse <-> B:Zebra dataset.

1st/2nd/3rd column is represented Original Image/Generated Image/Reconstructed Image.

|Origin: Horse / Gen: Zebra / Rec: Horse|
|:---:|
|![](imgs/CycleGAN_HZH_result.png)|

|Origin: Zebra / Gen: Horse / Rec: Zebra|
|:---:|
|![](imgs/CycleGAN_ZHZ_result.png)|

In my experience, 4GB per one batch size.

In the paper, there are 200 epochs.

With a NVIDIA RTX 2060, consume approximately 12 minutes per one epoch. (40 hours for 200 epochs)

#### Acknowledge

Code is implemented based on the paper and Official Pytorch Code version of CycleGAN. 

All credits goes to the authos of CycleGAN.

```
@inproceedings{CycleGAN2017,
  title={Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks},
  author={Zhu, Jun-Yan and Park, Taesung and Isola, Phillip and Efros, Alexei A},
  booktitle={Computer Vision (ICCV), 2017 IEEE International Conference on},
  year={2017}
}
```


## License

This project is licensed under the GPL v3 License. 














