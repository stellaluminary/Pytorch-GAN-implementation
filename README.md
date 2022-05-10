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
├── configs      # 
├── data
├── datasets    
├── model
├── results
├── utils
```

## Implementations

### CycleGAN

<b>Title of Paper</b> : Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks

<b>Authors</b> : Jun-Yan Zhu(first author), Taesung Park(first author), Phillip Isola, Alexei A. Efros

[[Offical Paper]](https://openaccess.thecvf.com/content_ICCV_2017/papers/Zhu_Unpaired_Image-To-Image_Translation_ICCV_2017_paper.pdf) [[Arxiv]](https://arxiv.org/abs/1703.10593)

[[Official Pytorch Code]](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)

#### Datasets

```
.
├── datasets                   
|   ├── <dataset_name>         # i.e. brucewayne2batman
|   |   ├── train              # Training
|   |   |   ├── A              # Contains domain A images (i.e. Bruce Wayne)
|   |   |   └── B              # Contains domain B images (i.e. Batman)
|   |   └── test               # Testing
|   |   |   ├── A              # Contains domain A images (i.e. Bruce Wayne)
|   |   |   └── B              # Contains domain B images (i.e. Batman)
```

First, download and setup a dataset.

Download method

1) Run below code file which is the same bash code of the [Official CycleGAN Pytorch Code](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/datasets/download_cyclegan_dataset.sh). (e.g. horse2zebra)

```
bash ./datasets/download_cyclegan_dataset.sh horse2zebra
```

2) Download from [UC Berkeley's repository](http://efrosgans.eecs.berkeley.edu/cyclegan/datasets)

Dataset will be placed in dataset folder

#### Train

```
$ python3 train.py --opt configs/cyclegan.yaml
```

#### Test

```
$ python3 test.py --opt configs/cyclegan.yaml
```

#### Pretrained model

later

#### Results

The model was trained on A:Horse <-> B: Zebra dataset.

1st/2nd/3rd column is represented Original Image/Generated Image/Reconstructed Image.

|Origin: Horse / Gen: Zebra / Rec: Horse|
|:---:|
|![]()|

|Origin: Zebra / Gen: Horse / Rec: Zebra|
|:---:|
|![]()|


#### Acknowledge

Code is referred to the paper and Official Pytorch Code version of authors of CycleGAN. 

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

This project is licensed under the GPL v3 License (https://en.wikipedia.org/wiki/GNU_General_Public_License)














